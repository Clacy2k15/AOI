# GUI 與 CLI 共用的完整推論主流程。輸入是使用者選定產品與一張影像，輸出是可追溯結果紀錄。
# 流程：載入兩模型 → RGB／前處理 → 各自推論 → 回映原圖 → OR 判決／定位 → NG 分類 → 保存。
# OK／NG 由偵測器決定；unknown 或分類失敗不會把已判 NG 的影像改成 OK。

"""One load/predict/output pipeline shared by the CLI and a future GUI."""
from __future__ import annotations
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from aoi.common import ROOT, configure_runtime, load_inference, write_json, output_path, artifact_identity
from aoi.fusion import combine, contour_image, qwen_localization, qwen_contour_image, QWEN_CONTOUR_RENDERING
from aoi.velm import VELMClient, classification_fingerprint


# 管理某個產品的已載入模型、正常參考圖、Qwen client 與結果保存。
# 同一實例可處理多張圖；切換產品應關閉舊實例，避免權重與門檻交叉使用。
class Pipeline:
    # 先啟用離線限制，核對產品及校正設定，再依一般／capsule ROI／pill ROI 路徑載入模型。
    # classify 控制分類；reuse_dev_results 僅供固定 dev 的已保存結果，不能拿來推論任意新圖。
    def __init__(self, config: dict, inference_path: str | Path | None = None, classify: bool = True,
                 reuse_dev_results: bool = False, split: str = 'dev'):
        from aoi.offline import enable_offline
        enable_offline()
        self.config = config
        self.product = config["project"]["category"]
        self.identity = artifact_identity(config, split)
        self.settings = load_inference(config, inference_path, require_calibrated=True)
        configure_runtime(config)
        os.environ["HF_HUB_OFFLINE"] = "1"
        import torch
        from aoi.inference import Detector, load_rgb

        torch.set_num_threads(config["project"]["cpu_threads"])
        cv2.setNumThreads(1)
        self.group = self.settings["selected_patchcore"]
        self.reference = load_rgb(ROOT / self.settings["normal_reference"])
        self.reuse_dev_results = reuse_dev_results
        self.classify_enabled = classify
        # toothbrush 的已採用規則只有 defective 一種瑕疵；NG 可直接映射，不送 Qwen 也不捏造分類信心。
        self.single_defect_class = ('defective' if classify and self.product == 'toothbrush'
            and set(self.settings['qwen']['labels'].values()) == {'defective', 'unknown_defect'} else None)
        self.efficientad = self.patchcore = None
        self.cached_classifications = {}
        capsule_pair = self.product == 'capsule' and self.settings.get('capsule_model_pair', {}).get('enabled')
        pill_patchcore = self.product == 'pill' and self.settings.get('pill_patchcore', {}).get('enabled')
        # ROI 模型的座標與分數配方已改變，不能把舊一般模型的 dev 快取當成新 ROI 推論結果。
        if (capsule_pair or pill_patchcore) and reuse_dev_results:
            raise ValueError('The adopted ROI detector requires live inference; legacy detector/classification caches are incompatible')
        # 這條支線只核對並重用開發成果；精簡發佈包沒有完整快取，新圖片應走下面的實際模型路徑。
        if reuse_dev_results:
            from aoi.inference import existing_dev_predictions
            from aoi.common import read_json
            self.cached_detectors = {}
            for name in ['efficientad', self.group]:
                arrays, metadata = existing_dev_predictions(config,name)
                self.cached_detectors[name] = {str(Path(row['path']).resolve()).casefold():
                    {**row,'score':float(arrays['scores'][i]),'map':arrays['maps'][i]}
                    for i,row in enumerate(metadata['rows'])}
            result_path = self.settings.get('dev_result_path')
            if classify and not self.single_defect_class:
                if not result_path:
                    raise ValueError('No selected dev classification result configured')
                saved = read_json(ROOT/result_path)
                if not saved['complete'] or saved['settings_fingerprint'] != classification_fingerprint(config,self.settings):
                    raise ValueError('Saved dev classification does not match current inference settings')
                self.cached_classifications = {r['image_sha256']:r for r in saved['results']}
        else:
            if capsule_pair:
                from aoi.capsule_runtime import CapsuleDetector
                self.efficientad = CapsuleDetector(config, 'efficientad', self.settings)
                try:
                    self.patchcore = CapsuleDetector(config, self.group, self.settings)
                # 若第二個模型載入失敗，先釋放已載入的第一個，避免建構失敗仍占住顯存。
                except Exception:
                    self.efficientad.close()
                    raise
            else:
                self.efficientad = Detector(config, "efficientad")
                try:
                    if pill_patchcore:
                        from aoi.pill_runtime import PillPatchcore
                        self.patchcore = PillPatchcore(config, self.settings)
                    else:
                        self.patchcore = Detector(config, self.group)
                except Exception:
                    self.efficientad.close()
                    raise
        self.client = VELMClient(self.settings["qwen"]) if classify and not reuse_dev_results and not self.single_defect_class else None

    # 對單張本機影像執行偵測、定位、分類與輸出，回傳包含狀態、分數、區域、檔案及耗時的 dict。
    # 每張圖都有獨立目錄；一般執行錯誤寫成 REVIEW 紀錄，方便批次繼續處理其他影像。
    def predict(self, input_path: str | Path, output_dir: str | Path | None = None) -> dict:
        from aoi.inference import load_rgb

        started = time.perf_counter()
        input_path = Path(input_path).resolve()
        # sample_id 依輸入絕對路徑及產品產生，主要用於隔離輸出目錄；它不是影像內容 SHA256。
        key = str(input_path) if self.product == "metal_nut" else self.product + ":" + str(input_path)
        sample_id = hashlib.sha256(key.encode()).hexdigest()[:16]
        output_dir = output_dir or output_path(self.config, "predictions")
        directory = ROOT / output_dir / sample_id
        directory.mkdir(parents=True, exist_ok=True)
        # 先初始化為 REVIEW／failed，只有後續步驟成功才逐項填入結果，避免未完成影像被誤認為 OK。
        record = {"sample_id": sample_id, "input_path": str(input_path), "product": self.product,
                  "identity": self.identity,
                  "patchcore_variant": self.group, "score_efficientad": None, "score_patchcore": None,
                  "classification_score": None, "classification_score_calibrated": False,
                  "classification_status": "failed", "defect_type": "unknown_defect", "decision": "REVIEW",
                  "localization_status": "failed", "region_count": 0, "regions": [],
                  "timings_ms": {}, "artifacts": {}, "error": None}
        try:
            io_start = time.perf_counter()
            rgb = load_rgb(input_path)
            record["timings_ms"]["input_read"] = (time.perf_counter() - io_start) * 1000
            # Every successfully processed frame runs both detectors, sequentially.
            digest = None
            if self.reuse_dev_results:
                key = str(input_path).casefold()
                # cache-only 額外核對影像位元組，即使檔名一樣，只要內容不同就不可沿用已保存的分類。
                digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
                if self.classify_enabled and not self.single_defect_class and digest not in self.cached_classifications:
                    raise ValueError('Input bytes do not match a saved fixed-dev classification sample')
                if key not in self.cached_detectors['efficientad']:
                    # Anonymous CLI fixtures may be byte-identical copies of fixed dev images.
                    saved = self.cached_classifications.get(digest)
                    if saved is None and self.single_defect_class:
                        match = next((p for p in self.cached_detectors['efficientad']
                            if hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest), None)
                        saved = {'input_path': match} if match else None
                    if saved is None:
                        raise ValueError('Input has no identical authorized dev result; cache-only mode refuses inference')
                    key = str(Path(saved['input_path']).resolve()).casefold()
                ea = dict(self.cached_detectors['efficientad'][key])
                pc = dict(self.cached_detectors[self.group][key])
                record['result_reuse'] = {'scope':'same product/config/checkpoint/threshold/prompt and identical dev image',
                    'detector_source_path':key,'classification_result':None if self.single_defect_class else self.settings.get('dev_result_path'),
                    'timings':'detector timing fields retain original actual dev inference; no new model inference'}
            else:
                ea = self.efficientad.predict(rgb)
                pc = self.patchcore.predict(rgb)
            # 保留兩個模型的原始分數，不把不同量尺的分數相加或假裝是同一種機率。
            record.update(score_efficientad=ea["score"], score_patchcore=pc["score"])
            record["preprocessing"] = {"efficientad": ea["metadata"], "patchcore": pc["metadata"]}
            for label, result in [("efficientad", ea), ("patchcore", pc)]:
                record["timings_ms"][label + "_preprocessing"] = result["preprocessing_ms"]
                record["timings_ms"][label] = result["inference_ms"]
                record["timings_ms"][label + "_remap"] = result["remap_ms"]
            fusion_start = time.perf_counter()
            # combine 產生對外 OR 判決與遮罩；qwen_localization 另外選紅圈來源，不反過來更改判決。
            fused = combine(ea, pc, self.settings["thresholds"])
            qloc = qwen_localization(ea, pc, self.settings['thresholds'],
                                     self.settings['qwen'].get('mask_source', 'OR'), fused)
            # 布林大陣列另存成 PNG／npz，不直接塞進 JSON；其餘可序列化欄位合併進 record。
            mask = fused.pop("mask")
            record.update(fused)
            contour = contour_image(rgb, mask)
            qwen_contour = qwen_contour_image(rgb, qloc['mask'], self.settings['qwen']['image_long_edge'])
            record.update(qwen_mask_source=qloc['source'], qwen_localization_status=qloc['localization_status'],
                          qwen_region_count=qloc['region_count'],
                          qwen_mask_policy=self.settings['qwen'].get('mask_source','OR'),
                          qwen_mask_fallback_reason=qloc.get('fallback_reason'),
                          qwen_contour_rendering={**QWEN_CONTOUR_RENDERING,
                                                 'image_size_hw': list(qwen_contour.shape[:2])})
            record["timings_ms"]["fusion"] = (time.perf_counter() - fusion_start) * 1000
            qwen_start = time.perf_counter()
            # 正常圖跳過分類；這個 good 來自兩個偵測器皆未達影像門檻，不是 Qwen 重新判定。
            if record["decision"] == "OK":
                record.update(defect_type="good", classification_status="skipped_ok")
            elif self.single_defect_class:
                record.update(defect_type=self.single_defect_class, classification_status='inferred_single_class',
                              classification_score_source='detector_ng_and_single_defect_class',
                              classification_score_calibrated=False, qwen_called=False)
            elif self.reuse_dev_results and self.classify_enabled:
                saved = self.cached_classifications[digest]
                for key in ['defect_type','classification_status','classification_score','classification_score_source',
                            'classification_score_calibrated','raw_final_answer','usage','error']:
                    record[key] = saved.get(key)
                record['result_reuse']['original_qwen_ms'] = saved['inference_ms']
            elif self.client:
                # Original RGB/reference; red contours are drawn at Qwen's input resolution.
                # 三圖依序是正常參考、原始 query、紅圈 query；即使沒有定位仍可提供完整 query 分類。
                classification = self.client.classify(self.reference, rgb, qwen_contour, qloc['localization_status'])
                response = classification.pop("backend_response", None)
                if response is not None:
                    response_path = directory / "qwen_response.json"
                    write_json(response_path, response)
                    record["artifacts"]["qwen_response"] = str(response_path)
                record.update(classification)
            else:
                record.update(classification_status="disabled", defect_type="unknown_defect")
            record["timings_ms"]["qwen"] = (time.perf_counter() - qwen_start) * 1000 if record["decision"] == "NG" else 0.0
            output_start = time.perf_counter()
            overlay = contour.copy()
            # bbox 的右／下界原本是不含端點，OpenCV rectangle 則包含端點，所以繪圖時使用 x2-1／y2-1。
            for region in record["regions"]:
                x1, y1, x2, y2 = region["bbox_xyxy"]
                cv2.rectangle(overlay, (x1, y1), (x2 - 1, y2 - 1), (255, 60, 0), 1)
            title = f"{record['decision']} | {record['defect_type']} | EA={ea['score']:.4f} PC={pc['score']:.4f} regions={record['region_count']}"
            if record["classification_score"] is not None:
                title += f" token_score={record['classification_score']:.4f} (uncalibrated)"
            cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 55), (15, 15, 15), -1)
            cv2.putText(overlay, title[:110], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .45,
                        (40, 255, 40) if record["decision"] == "OK" else (255, 100, 100), 1)
            cv2.putText(overlay, "localization=" + record["localization_status"], (8, 44), cv2.FONT_HERSHEY_SIMPLEX, .45, (230, 230, 230), 1)
            # OpenCV imwrite 需要 BGR；內部處理保留 RGB，因此僅在存 PNG 的邊界轉換通道順序。
            for name, image in [("contour", contour), ("overlay", overlay)]:
                path = directory / (name + ".png")
                if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                    raise IOError("Could not save " + str(path))
                record["artifacts"][name] = str(path)
            path = directory / 'qwen_contour.png'
            if not cv2.imwrite(str(path), cv2.cvtColor(qwen_contour, cv2.COLOR_RGB2BGR)):
                raise IOError('Could not save actual Qwen contour')
            record['artifacts']['qwen_contour'] = str(path)
            mask_path = directory / "mask.png"
            if not cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255):
                raise IOError("Could not save mask")
            record["artifacts"]["mask"] = str(mask_path)
            # Raw maps allow stage C to reuse predictions for the identical sample/settings.
            map_path = directory / "anomaly_maps.npz"
            # 保存連續原始分數圖，讓後續分析可區分模型輸出與門檻二值化效果。
            np.savez_compressed(map_path, efficientad=ea["map"], patchcore=pc["map"])
            record["artifacts"]["anomaly_maps"] = str(map_path)
            record["timings_ms"]["output"] = (time.perf_counter() - output_start) * 1000
        # 偵測／定位／寫檔的例外會轉為 REVIEW 並附上錯誤；Qwen 自身失敗由 client 回傳 failed，仍保留 NG。
        except Exception as error:
            record.update(decision="REVIEW", defect_type="unknown_defect", classification_status="failed",
                          error=f"{type(error).__name__}: {error}")
        record["timings_ms"]["total"] = (time.perf_counter() - started) * 1000
        record["artifacts"]["json"] = str(directory / "result.json")
        # 單張 JSON 在 finally 之外統一保存；若此最後寫入也失敗，例外會向外傳遞，不能宣稱成果已落盤。
        write_json(directory / "result.json", record)
        return record

    # 釋放已建立的 EfficientAD 與 PatchCore adapter。
    # Qwen 是外部共用程序，不在此任意終止；程序管理交由啟動／停止工具核對身分後處理。
    def close(self):
        if self.efficientad is not None:
            self.efficientad.close()
        if self.patchcore is not None:
            self.patchcore.close()


# 保存整批完整 results.json 及便於表格分析的 results.csv。
# CSV 使用 utf-8-sig 讓 Excel 辨識中文；多個 bbox 以 JSON 字串放在單一欄位中。
def save_records(records: list[dict], output_dir: str | Path) -> None:
    directory = ROOT / output_dir
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "results.json", {"results": records})
    columns = ["product", "sample_id", "input_path", "patchcore_variant", "decision", "defect_type", "score_efficientad",
               "score_patchcore", "classification_score", "classification_status", "localization_status", "region_count", "error"]
    with open(directory / "results.csv", "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns + ["qwen_ms", "total_ms", "bbox_xyxy"])
        writer.writeheader()
        for record in records:
            row = {key: record.get(key) for key in columns}
            row.update(qwen_ms=record["timings_ms"].get("qwen"), total_ms=record["timings_ms"]["total"],
                       bbox_xyxy=json.dumps([region["bbox_xyxy"] for region in record["regions"]]))
            writer.writerow(row)
