"""本機離線展示：選產品、輸入影像，查看 OK／NG 與 mask。"""
from __future__ import annotations
import argparse
import atexit
from datetime import datetime
from pathlib import Path
import sys
import threading
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aoi.offline import enable_offline
enable_offline()
from aoi.common import ROOT, load_config, read_json
from aoi.pipeline import Pipeline, save_records
PRODUCTS = read_json(ROOT / 'outputs/expansion/product_registry.json')['products']
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


class DemoRunner:
    def __init__(self):
        self.pipeline = None
        self.product = None
        self.lock = threading.Lock()

    def close(self):
        if self.pipeline is not None:
            self.pipeline.close()
            self.pipeline = None
        self.product = None

    def run(self, product, upload, input_path):
        import gradio as gr
        with self.lock:
            if product not in PRODUCTS:
                raise gr.Error('請先選擇產品。')
            source = (Path(input_path.strip().strip('"')) if input_path and input_path.strip()
                      else Path(upload) if upload else None)
            paths = ([source] if source and source.is_file() and source.suffix.lower() in EXTENSIONS else
                     sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
                     if source and source.is_dir() else [])
            if not paths:
                raise gr.Error('請上傳支援的影像，或填入本機影像／資料夾路徑。')
            if self.product != product:
                self.close()
                self.pipeline = Pipeline(load_config(PRODUCTS[product]['config']))
                self.product = product
            destination = ROOT / 'outputs/demo' / datetime.now().strftime('%Y%m%d_%H%M%S_%f') / product
            records = [self.pipeline.predict(path, destination) for path in paths]
            save_records(records, destination)
            overlays = [(r['artifacts']['overlay'], f"{Path(r['input_path']).name} | {r['decision']}")
                        for r in records[:12] if 'overlay' in r['artifacts']]
            masks = [(r['artifacts']['mask'], f"{Path(r['input_path']).name} | {r['decision']}")
                     for r in records[:12] if 'mask' in r['artifacts']]
            rows = [[Path(r['input_path']).name, r['decision'] or '執行失敗', r['error']] for r in records]
            archive = destination / 'results.zip'
            with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as output:
                for path in destination.rglob('*'):
                    if path.is_file() and path.suffix in {'.png', '.json', '.csv'}:
                        output.write(path, path.relative_to(destination))
            counts = {d: sum(r['decision'] == d for r in records) for d in ('OK', 'NG')}
            errors = sum(r['error'] is not None for r in records)
            message = f"完成 {len(records)} 張：OK {counts['OK']}、NG {counts['NG']}、執行失敗 {errors}。"
            return message, overlays, masks, rows, [str(destination / 'results.json'),
                                                   str(destination / 'results.csv'), str(archive)]


def build_demo():
    import gradio as gr
    runner = DemoRunner()
    atexit.register(runner.close)
    with gr.Blocks(title='AOI OK／NG 與 mask') as demo:
        gr.Markdown('# AOI OK／NG 與 mask\n選擇產品後，上傳影像或填入本機資料夾路徑。')
        with gr.Row():
            product = gr.Dropdown(choices=list(PRODUCTS), value='metal_nut', label='產品')
            upload = gr.File(type='filepath', file_types=['image'], label='單張影像')
        path = gr.Textbox(label='本機影像／資料夾路徑（填寫時優先使用）')
        run = gr.Button('開始檢測', variant='primary')
        status = gr.Textbox(label='執行結果', interactive=False)
        overlays = gr.Gallery(label='檢測疊圖（前 12 張）', columns=3)
        masks = gr.Gallery(label='Mask：白色為異常候選區域（前 12 張）', columns=3)
        table = gr.Dataframe(headers=['影像', '判決', '錯誤'], interactive=False)
        files = gr.File(label='下載結果與全部 mask／疊圖', file_count='multiple')
        run.click(runner.run, inputs=[product, upload, path],
                  outputs=[status, overlays, masks, table, files],
                  concurrency_limit=1, concurrency_id='aoi_gpu', api_name='predict')
    return demo.queue(default_concurrency_limit=1), runner


def main():
    import gradio as gr
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    demo, runner = build_demo()
    theme = gr.themes.Base(font=['Arial', 'sans-serif'], font_mono=['Consolas', 'monospace'])
    try:
        demo.launch(server_name='127.0.0.1', server_port=args.port, share=False, inbrowser=False,
                    allowed_paths=[str(ROOT / 'outputs')], theme=theme)
    finally:
        runner.close()


if __name__ == '__main__':
    main()
