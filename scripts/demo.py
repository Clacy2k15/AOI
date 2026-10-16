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
PRODUCTS = read_json(ROOT/'outputs/expansion/product_registry.json')['products']
TIMING_HEADERS = ['影像', '讀圖', '前處理', 'EfficientAD 推論', 'PatchCore 推論',
                  '回映與定位', '瑕疵分類', '結果保存', '單張總耗時']


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


def timing_row(record):
    timings = record['timings_ms']

    def measured(*keys):
        values = [timings[key] for key in keys if timings.get(key) is not None]
        return round(sum(values), 2) if values else None

    return [Path(record['input_path']).name, measured('input_read'),
            measured('efficientad_preprocessing', 'patchcore_preprocessing'),
            measured('efficientad'), measured('patchcore'),
            measured('efficientad_remap', 'patchcore_remap', 'fusion'),
            measured('qwen'), measured('output'), measured('total')]


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


class DemoRunner:
    def __init__(self):
        self.pipeline = None
        self.product = None
        self.lock = threading.Lock()

    def close(self):
        if self.pipeline:
            self.pipeline.close()
            self.pipeline = None
        self.product = None

    def run(self, product, upload, input_path):
        import gradio as gr
        from integrate_products import ensure_server
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
                if product != 'toothbrush':
                    ensure_server(inference_path(config))
                self.pipeline = Pipeline(config, split='demonstration')
                self.product = product
            destination = ROOT/'outputs/demo'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')/product
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
            gallery = [(r['artifacts']['overlay'], f"{Path(r['input_path']).name} | {r['decision']} | {r['defect_type']}")
                       for r in records[:12] if 'overlay' in r['artifacts']]
            archive = destination/'results.zip'
            with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
                for file in destination.rglob('*'):
                    if file.is_file() and file.suffix in {'.png', '.json', '.csv'}:
                        z.write(file, file.relative_to(destination))
            counts = {d: sum(r['decision'] == d for r in records) for d in ['OK', 'NG', 'REVIEW']}
            message = f"完成 {len(records)} 張：OK {counts['OK']}、NG {counts['NG']}、REVIEW {counts['REVIEW']}。結果：{destination}"
            return message, gallery, rows, timing_rows, [str(destination/'results.json'),
                    str(destination/'results.csv'), str(destination/'timings.csv'), str(archive)]


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


def main():
    import gradio as gr
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    demo, runner = build_demo()
    # System fonts keep the page usable with the browser completely offline.
    theme = gr.themes.Base(font=['Arial', 'sans-serif'], font_mono=['Consolas', 'monospace'])
    try:
        demo.launch(server_name='127.0.0.1', server_port=args.port, share=False, inbrowser=False,
                    allowed_paths=[str(ROOT/'outputs')], theme=theme)
    finally:
        runner.close()


if __name__ == '__main__':
    main()
