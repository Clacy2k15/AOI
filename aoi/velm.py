"""Local llama-server HTTP adapter. Vision/tokenization stay in llama.cpp."""
from __future__ import annotations
import base64
import json
import math
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from aoi.common import ROOT, read_json
from aoi.fusion import QWEN_CONTOUR_RENDERING, resize_qwen_image


def classification_fingerprint(config: dict, settings: dict) -> str:
    import hashlib
    from aoi.common import artifact_identity
    weights = {}
    for key in ['model_path', 'mmproj_path']:
        path = (ROOT/settings['qwen'][key]).resolve()
        stat = path.stat()
        weights[key] = {'path':str(path),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}
    value = {'identity':artifact_identity(config),'qwen':settings['qwen'],'weights':weights,
             'thresholds':settings['thresholds'],'selected_patchcore':settings['selected_patchcore'],
             'normal_reference':settings['normal_reference'],
             'contour_rendering':QWEN_CONTOUR_RENDERING}
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

LABELS = {"1": "bent", "2": "color", "3": "flip", "4": "scratch", "5": "unknown_defect"}
SYSTEM_PROMPT = "Classify visible anomalies in industrial metal nuts using the supplied images and class definitions.\nYour final answer must contain exactly one digit from 1 to 5, with no other text."
CLASS_PROMPT = """The detector may produce false alarms. Ignore the red annotation when judging physical color, shape, and surface marks.
If localization is missing, Image 3 has no contours. Inspect the full query image.
Compare the original query with the normal reference and consider the full nut as well as the proposed regions.

Choose one class:
1 = bent: visible deformation of the nut body, outline, edge, or opening.
2 = color: abnormal surface color, discoloration, or color patches visible in the original query.
3 = flip: the nut presents the opposite face compared with the expected normal face; an ordinary in-plane rotation alone is not sufficient.
4 = scratch: visible scratches, scoring, or abrasion marks on the nut surface.
5 = unknown_defect: no clear visible defect, insufficient evidence, an unsupported defect, or multiple equally plausible classes.

Your final answer must contain exactly one digit from 1 to 5, with no other text."""


def label_mapping(classification: dict | None = None) -> dict:
    labels = {str(k): v for k, v in (classification or {}).get("labels", LABELS).items()}
    if not labels or any(not k.isascii() or not k.isdecimal() or str(int(k)) != k for k in labels):
        raise ValueError("Class IDs must be complete positive decimal strings")
    if sorted(map(int, labels)) != list(range(1, len(labels) + 1)) or list(labels.values()).count("unknown_defect") != 1:
        raise ValueError("Expected contiguous IDs and exactly one unknown_defect")
    return labels


def output_grammar(labels: dict) -> str:
    return 'root ::= ' + ' | '.join(json.dumps(k) for k in sorted(labels, key=int))


def system_prompt(product: str, labels: dict) -> str:
    if product == "metal_nut" and labels == LABELS:
        return SYSTEM_PROMPT
    return (f"Classify visible anomalies in product {product} using the supplied images and class definitions.\n"
            "Return exactly one class ID from " + ", ".join(sorted(labels, key=int)) + ", with no other text.")


def product_options(options: dict, product: str, classification: dict | None = None) -> dict:
    options = dict(options)
    labels = label_mapping(classification)
    if product != "metal_nut" and classification is None:
        raise ValueError("New products require their own classification mapping")
    if classification and classification.get("prompt") is not None:
        prompt = classification["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Explicit classification prompt must be nonempty text")
        prompt = prompt.strip()
    elif classification:
        prompt = ("The detector may produce false alarms. Ignore the red annotation when judging physical appearance.\n"
                  "Predicted locations may be wrong or missing; inspect the full original query and normal reference.\n")
        if classification.get("instructions"):
            prompt += classification["instructions"].strip() + "\n"
        prompt += f"Normal appearance: {classification['normal']}\n\nChoose one class:\n"
        descriptions = classification["defects"]
        for key in sorted(labels, key=int):
            label = labels[key]
            description = ("No clear defect, insufficient evidence, unsupported defect, or ambiguous class."
                           if label == "unknown_defect" else descriptions[label])
            prompt += f"{key} = {label}: {description}\n"
        prompt += "\nReturn exactly one complete class ID from " + ", ".join(sorted(labels, key=int)) + ", with no other text."
    else:
        prompt = options.get("class_prompt") or CLASS_PROMPT
    options.update(product=product, labels=labels, class_prompt=prompt, output_grammar=output_grammar(labels),
                   system_prompt=system_prompt(product, labels))
    return options


def classification_content(reference, query, contour, localization_status, image_item, class_prompt=CLASS_PROMPT,
                           product="metal_nut"):
    """Share the exact image order and instructions with the vision fine-tune."""
    reference_text = "metal nut" if product == "metal_nut" else product
    return [{"type": "text", "text": f"Product: {product}.\n\nImage 1 is a normal reference {reference_text}."},
            image_item(reference),
            {"type": "text", "text": "Image 2 is the original query image."}, image_item(query),
            {"type": "text", "text": "Image 3 is the same query image with red contours around regions proposed by the anomaly detectors."},
            image_item(contour),
            {"type": "text", "text": "Localization status: " + localization_status + ".\n" + class_prompt}]


def parse_response(response: dict, labels: dict | None = None) -> dict:
    labels = label_mapping({"labels": labels}) if labels is not None else LABELS
    choice = response["choices"][0]
    answer = (choice["message"].get("content") or "").strip()
    if choice.get("finish_reason") == "length" or answer not in labels:
        raise ValueError("Invalid/truncated final class answer: " + repr(answer))
    score, source = None, None
    probabilities = (choice.get("logprobs") or {}).get("content", [])
    for token in probabilities:
        if (token.get("token") or "").strip() == answer and token.get("logprob") is not None:
            score, source = math.exp(float(token["logprob"])), "final_label_token_logprob"
            break
    if score is None:
        for token in response.get("completion_probabilities", []):
            if (token.get("tok_str") or token.get("content") or "").strip() == answer:
                if "prob" in token:
                    score, source = float(token["prob"]), "final_label_token_probability"
                    break
    return {"defect_type": labels[answer], "classification_status": "unknown" if labels[answer] == "unknown_defect" else "classified",
            "classification_score": score, "classification_score_source": source,
            "classification_score_calibrated": False, "raw_final_answer": answer,
            "usage": response.get("usage"), "backend_response": response, "error": None}


def recorded_server_command(record: dict) -> str:
    """Read the recorded Windows process command, guarding stale/unrelated PIDs."""
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, record["pid"])
    if not handle:
        if ctypes.get_last_error() == 87:
            raise RuntimeError("Recorded Qwen PID is not running")
        raise RuntimeError("Cannot access recorded Qwen PID; do not assume it stopped")
    try:
        query = ctypes.WinDLL("ntdll").NtQueryInformationProcess
        query.argtypes = [wintypes.HANDLE, wintypes.ULONG, ctypes.c_void_p, wintypes.ULONG,
                          ctypes.POINTER(wintypes.ULONG)]
        size = wintypes.ULONG()
        query(handle, 60, None, 0, ctypes.byref(size))
        if not size.value:
            raise RuntimeError("Unable to read Qwen process command line")
        buffer = ctypes.create_string_buffer(size.value)
        if query(handle, 60, buffer, size.value, ctypes.byref(size)) != 0:
            raise RuntimeError("Unable to verify Qwen process command line")
        class UnicodeString(ctypes.Structure):
            _fields_ = [("length", wintypes.USHORT), ("maximum", wintypes.USHORT), ("data", ctypes.c_void_p)]
        value = UnicodeString.from_buffer(buffer)
        command = ctypes.wstring_at(value.data, value.length // 2)
        # Use Windows' existing argv parser, then compare complete path arguments.
        shell = ctypes.WinDLL("shell32")
        shell.CommandLineToArgvW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
        shell.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
        count = ctypes.c_int()
        pointer = shell.CommandLineToArgvW(command,ctypes.byref(count))
        if not pointer:
            raise RuntimeError("Unable to parse recorded Qwen process command")
        try:
            argv = [pointer[i] for i in range(count.value)]
        finally:
            kernel.LocalFree.argtypes = [ctypes.c_void_p]
            kernel.LocalFree(pointer)
        if Path(argv[0]).resolve() != Path(record['executable']).resolve():
            raise RuntimeError("Recorded Qwen PID has a different executable")
        for flag in ["--model","--mmproj"]:
            path = record["command"][record["command"].index(flag)+1]
            if flag not in argv or Path(argv[argv.index(flag)+1]).resolve() != Path(path).resolve():
                raise RuntimeError("Recorded Qwen PID was reused or has different weights")
        return command
    finally:
        kernel.CloseHandle(handle)


def verify_server_model(options: dict) -> None:
    record = read_json(ROOT / "outputs/qwen_server_process.json")
    command = recorded_server_command(record)
    for key, flag in [("model_path", "--model"), ("mmproj_path", "--mmproj")]:
        expected = str((ROOT / options[key]).resolve())
        recorded = record["command"][record["command"].index(flag) + 1]
        if Path(recorded).resolve() != Path(expected) or expected.casefold() not in command.casefold():
            raise RuntimeError("Qwen weights differ from product config; stop recorded server and start the correct pair")
    with urllib.request.urlopen(options["base_url"] + "/props", timeout=10) as stream:
        properties = json.load(stream)
    if Path(properties.get("model_path", "")).resolve() != (ROOT / options["model_path"]).resolve():
        raise RuntimeError("The responding Qwen server has different GGUF weights")


def stop_recorded_server(record: dict) -> None:
    """Terminate only a verified project PID and wait for its actual exit."""
    import ctypes
    import os
    import signal
    from ctypes import wintypes
    recorded_server_command(record)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE,wintypes.DWORD]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x100000,False,record['pid'])
    if not handle:
        raise RuntimeError('Unable to wait on the recorded Qwen process')
    try:
        os.kill(record['pid'],signal.SIGTERM)
        if kernel.WaitForSingleObject(handle,30000) != 0:
            raise RuntimeError('Recorded Qwen server has not exited; do not start another GPU job')
    finally:
        kernel.CloseHandle(handle)


class VELMClient:
    def __init__(self, options: dict):
        self.options = options
        if options.get("product","metal_nut") != "metal_nut" and (not options.get("labels") or not options.get("class_prompt")):
            raise ValueError("Product Qwen options require resolved labels and class descriptions")
        self.labels = label_mapping({"labels": options.get("labels", LABELS)})
        self._server_checked = False
        endpoint = urlparse(options["base_url"])
        if endpoint.hostname not in {"127.0.0.1", "localhost", "::1"} or endpoint.scheme != "http":
            raise ValueError("Classification endpoint must be local HTTP")

    def _image(self, rgb) -> dict:
        import cv2
        image = resize_qwen_image(rgb, self.options["image_long_edge"])
        success, png = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not success:
            raise ValueError("OpenCV could not encode PNG")
        return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode("ascii")}}

    def classify(self, reference, query, contour, localization_status: str) -> dict:
        content = classification_content(reference, query, contour, localization_status, self._image,
                                         self.options.get("class_prompt", CLASS_PROMPT),
                                         product=self.options.get("product", "metal_nut"))
        options = self.options
        payload = {"model": "aoi-qwen", "messages": [{"role": "system", "content": options.get("system_prompt", SYSTEM_PROMPT)},
                                                      {"role": "user", "content": content}],
                   "chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none",
                   "max_tokens": options["max_tokens"], "temperature": options["temperature"],
                   "top_p": options["top_p"], "top_k": options["top_k"], "min_p": options["min_p"],
                   "presence_penalty": options["presence_penalty"], "repeat_penalty": options["repeat_penalty"],
                   "seed": options["seed"], "stream": False, "logprobs": True, "top_logprobs": len(self.labels),
                   "n_probs": len(self.labels), "post_sampling_probs": False,
                   "grammar": output_grammar(self.labels)}
        try:
            if not self._server_checked:
                verify_server_model(options)
                self._server_checked = True
            request = urllib.request.Request(options["base_url"] + "/v1/chat/completions",
                                             data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=options["request_timeout_seconds"]) as stream:
                response = json.load(stream)
        except Exception as error:
            return {"defect_type": "unknown_defect", "classification_status": "failed", "classification_score": None,
                    "classification_score_source": None, "classification_score_calibrated": False,
                    "error": f"{type(error).__name__}: {error}"}
        try:
            return parse_response(response, self.labels)
        except Exception as error:
            # Keep a received answer for diagnosis without another model request.
            return {"defect_type": "unknown_defect", "classification_status": "failed", "classification_score": None,
                    "classification_score_source": None, "classification_score_calibrated": False,
                    "backend_response": response, "error": f"{type(error).__name__}: {error}"}
