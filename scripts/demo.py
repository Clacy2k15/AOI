# Gradio 本機展示介面：選產品、上傳單張影像或輸入本機路徑，顯示推論與可下載成果。
# GUI 與 CLI 共用 Pipeline，不各自維護不同的門檻、前處理或判決邏輯。
# 介面只綁定 localhost，關閉分享並使用系統字型；初次安裝後可斷網執行。

"""Offline Gradio product selector over the same Pipeline used by predict.py."""
from __future__ import annotations
import argparse
import atexit
import csv
from datetime import datetime
from pathlib import Path
import sys
import threading
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aoi.offline import enable_offline
enable_offline()
from aoi.common import ROOT, load_config, read_json, inference_path
from aoi.pipeline import Pipeline, save_records

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
# 精簡 registry 只列可選產品及其設定位置，不從任意輸入圖片自動推斷產品種類。
PRODUCTS = read_json(ROOT/'outputs/expansion/product_registry.json')['products']
TIMING_HEADERS = ['影像', '讀圖', '前處理', 'EfficientAD 推論', 'PatchCore 推論',
                  '回映與定位', '瑕疵分類', '結果保存', '單張總耗時']


# 讀取所選產品目前有效的 image 門檻並組成介面說明文字。
# 只解釋兩模型 OR 的 OK／NG 規則，不把像素定位門檻或 Qwen 信心混成判決門檻。
def decision_criteria(product):
    if product not in PRODUCTS:
        return '請先選擇產品以查看 OK／NG 判定標準。'
    config = load_config(PRODUCTS[product]['config'])
    settings = load_config(inference_path(config))
    ea = settings['thresholds']['efficientad']['image']
    pc = settings['thresholds']['patchcore']['image']
    return (f'**{product} 的 OK／NG 判定標準**\n\n'
            f'- **NG（異常）**：EfficientAD 分數 ≥ `{ea}`，或 PatchCore 分數 ≥ `{pc}`，任一條件成立即判為 NG。\n'
            f'- **OK（正常）**：EfficientAD 分數 < `{ea}`，且 PatchCore 分數 < `{pc}`，兩者皆低於門檻才判為 OK。\n'
            '- Qwen（VLM）負責 NG 影像的瑕疵分類；OK／NG 由上述影像異常分數判定。')


# 把單張紀錄的 timings_ms 轉成固定欄位順序，供 GUI 表格與 timings.csv 共用。
# 前處理／回映欄位會加總兩個模型的對應耗時，缺少測量值時保留 None。
def timing_row(record):
    timings = record['timings_ms']

    # 加總指定且非 None 的毫秒欄位，再四捨五入到兩位小數。
    # 完全沒有量測值時回傳 None，避免把缺值偽裝成 0 ms。
    def measured(*keys):
        values = [timings[key] for key in keys if timings.get(key) is not None]
        return round(sum(values), 2) if values else None

    return [Path(record['input_path']).name, measured('input_read'),
            measured('efficientad_preprocessing', 'patchcore_preprocessing'),
            measured('efficientad'), measured('patchcore'),
            measured('efficientad_remap', 'patchcore_remap', 'fusion'),
            measured('qwen'), measured('output'), measured('total')]


# 在已保存的 overlay 圖頂部重畫分數與分類信心說明，讓下載圖與展示頁一致。
# 優先使用 Windows 微軟正黑體，找不到時回退預設字型；此步不改動模型或遮罩。
def format_demo_overlay(record):
    """Use the demonstration's score label on the saved preview as well."""
    path = record.get('artifacts', {}).get('overlay')
    if not path:
        return
    from PIL import Image, ImageDraw, ImageFont
    image = Image.open(path).convert('RGB')
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 54), fill=(15, 15, 15))
    font_path = Path('C:/Windows/Fonts/msjh.ttc')
    font = ImageFont.truetype(str(font_path), 14) if font_path.is_file() else ImageFont.load_default()
    first = (f"{record['decision']} | {record['defect_type']} | "
             f"EA={record['score_efficientad']:.4f} PC={record['score_patchcore']:.4f} "
             f"regions={record['region_count']}")
    score = record['classification_score']
    confidence = '—' if score is None else f'{score:.4f}'
    second = f"Qwen（VLM）瑕疵分類信心：{confidence} | localization={record['localization_status']}"
    color = (40, 255, 40) if record['decision'] == 'OK' else (255, 100, 100)
    draw.text((8, 3), first, font=font, fill=color)
    draw.text((8, 29), second, font=font, fill=(230, 230, 230))
    image.save(path)


# 管理 GUI 的單一產品 pipeline 與執行鎖，避免多個請求同時操作同一組模型。
# 切換產品時先釋放舊模型，相同產品則重用已載入實例。
class DemoRunner:
    # 初始化尚未載入的 pipeline、產品名稱與 thread lock。
    # 模型採使用時才載入，單純開啟網頁不會立即占用兩個偵測器的顯存。
    def __init__(self):
        self.pipeline = None
        self.product = None
        self.lock = threading.Lock()

    # 關閉現有 pipeline 並重設產品狀態。
    # 同時註冊於程序退出流程，使正常離開 GUI 時也能釋放偵測器資源。
    def close(self):
        if self.pipeline:
            self.pipeline.close()
            self.pipeline = None
        self.product = None

    # 處理一次使用者提交：核對產品與路徑、必要時切換模型，逐張推論後整理表格、圖庫與下載檔。
    # 本機路徑有填值時優先於上傳檔；批次結果各有含微秒時間戳的輸出目錄。
    def run(self, product, upload, input_path):
        import gradio as gr
        from integrate_products import ensure_server
        # 這把鎖保護模型切換及整次批次執行，防止兩個使用者請求交錯釋放或覆用同一個 pipeline。
        with self.lock:
            if product not in PRODUCTS:
                raise gr.Error('請先選擇產品。')
            source = Path(input_path.strip().strip('"')) if input_path and input_path.strip() else Path(upload) if upload else None
            if source is None or not source.exists():
                raise gr.Error('請上傳影像，或填入本機影像／資料夾路徑。')
            paths = [source] if source.is_file() and source.suffix.lower() in EXTENSIONS else (
                sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS) if source.is_dir() else [])
            if not paths:
                raise gr.Error('找不到支援的影像。')
            config = load_config(PRODUCTS[product]['config'])
            if self.product != product:
                self.close()
                # 牙刷沿用單一瑕疵分類規則，不需要為這個產品啟動 Qwen。
                if product != 'toothbrush':
                    ensure_server(inference_path(config))
                self.pipeline = Pipeline(config, split='demonstration')
                self.product = product
            destination = ROOT/'outputs/demo'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')/product
            # 每張圖完整保存成功或 REVIEW 紀錄；畫面只摘要顯示，不丟棄錯誤影像。
            records = [self.pipeline.predict(path, destination) for path in paths]
            for record in records:
                format_demo_overlay(record)
            save_records(records, destination)
            timing_rows = [timing_row(record) for record in records]
            with (destination/'timings.csv').open('w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.writer(stream)
                writer.writerow([TIMING_HEADERS[0]]+[name+' (ms)' for name in TIMING_HEADERS[1:]])
                writer.writerows(timing_rows)
            rows = [[Path(r['input_path']).name, r['decision'], r['defect_type'], r['score_efficientad'],
                     r['score_patchcore'], r['classification_score'], r['region_count'],
                     str([v['bbox_xyxy'] for v in r['regions']]), r['classification_status'], r['error']]
                    for r in records]
            # 圖庫最多呈現前 12 張以控制頁面大小；完整批次仍在輸出與下載檔中保留。
            gallery = [(r['artifacts']['overlay'], f"{Path(r['input_path']).name} | {r['decision']} | {r['defect_type']}")
                       for r in records[:12] if 'overlay' in r['artifacts']]
            archive = destination/'results.zip'
            # ZIP 只收集 PNG／JSON／CSV，不把壓縮檔自己遞迴放進去，也不把大型 npz 放進展示下載包。
            with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
                for file in destination.rglob('*'):
                    if file.is_file() and file.suffix in {'.png', '.json', '.csv'}:
                        z.write(file, file.relative_to(destination))
            counts = {d: sum(r['decision'] == d for r in records) for d in ['OK', 'NG', 'REVIEW']}
            message = f"完成 {len(records)} 張：OK {counts['OK']}、NG {counts['NG']}、REVIEW {counts['REVIEW']}。結果：{destination}"
            return message, gallery, rows, timing_rows, [str(destination/'results.json'),
                    str(destination/'results.csv'), str(destination/'timings.csv'), str(archive)]


# 建立產品選單、門檻說明、影像輸入、結果／耗時表與成果下載元件，再綁定事件。
# 推論事件併發數固定 1；產品切換只更新門檻文字，不立刻執行模型。
def build_demo():
    import gradio as gr
    runner = DemoRunner()
    atexit.register(runner.close)
    with gr.Blocks(title='AOI 離線瑕疵檢測') as demo:
        gr.Markdown('# AOI 離線瑕疵檢測\n先選產品，再上傳單張影像或填入本機資料夾路徑。')
        with gr.Row():
            product = gr.Dropdown(choices=list(PRODUCTS), value='metal_nut', label='產品')
            upload = gr.File(type='filepath', file_types=['image'], label='單張影像')
        criteria = gr.Markdown(decision_criteria('metal_nut'))
        # 產品選單變更事件與正式推論事件分開，避免使用者只是切換選項就觸發模型計算。
        product.change(decision_criteria, inputs=product, outputs=criteria, queue=False,
                       api_name='decision_criteria')
        path = gr.Textbox(label='本機影像／資料夾路徑（填寫時優先使用）')
        run = gr.Button('開始檢測', variant='primary')
        status = gr.Textbox(label='執行結果', interactive=False)
        gallery = gr.Gallery(label='檢測疊圖（最多顯示前 12 張；全部保存於下載檔）', columns=3)
        gr.Markdown('EA／PC 為模型原始異常分數。模型信心是指 Qwen（VLM）對瑕疵類別的分類信心。紅圈只標示疑似區域，OK／NG 依影像異常分數判定。')
        table = gr.Dataframe(headers=['影像', '判決', '類型', 'EA 分數', 'PC 分數', '模型信心（Qwen／VLM）', '區域數', '座標 xyxy', '分類狀態', '錯誤'], interactive=False)
        gr.Markdown('### 各階段耗時（毫秒 ms）')
        timings = gr.Dataframe(headers=TIMING_HEADERS, interactive=False)
        files = gr.File(label='下載 JSON／CSV／疊圖與遮罩壓縮檔', file_count='multiple')
        run.click(runner.run, inputs=[product, upload, path], outputs=[status, gallery, table, timings, files],
                  concurrency_limit=1, concurrency_id='aoi_gpu', api_name='predict')
    return demo.queue(default_concurrency_limit=1), runner


# 解析 GUI 埠號，建立介面並在本機啟動 Gradio。
# share=False 避免公開分享；允許下載的專案檔案範圍限於 outputs，結束時關閉 runner。
def main():
    import gradio as gr
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    demo, runner = build_demo()
    # System fonts keep the page usable with the browser completely offline.
    # 使用已安裝的 Arial／Consolas，不依賴網路字型服務。
    theme = gr.themes.Base(font=['Arial', 'sans-serif'], font_mono=['Consolas', 'monospace'])
    try:
        demo.launch(server_name='127.0.0.1', server_port=args.port, share=False, inbrowser=False,
                    allowed_paths=[str(ROOT/'outputs')], theme=theme)
    finally:
        runner.close()


if __name__ == '__main__':
    main()
