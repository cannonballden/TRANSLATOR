import json, pathlib
from server.analyzer import analyze_media

EX_DIR = pathlib.Path(__file__).resolve().parents[1] / "examples"
GT = json.loads(pathlib.Path(__file__).with_name("ground_truth.json").read_text())

def iou_1d(a,b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1]-a[0]) + (b[1]-b[0]) - inter + 1e-9
    return inter/union

tp=fp=fn=0
for case in GT:
    f = str(EX_DIR / case["file"])
    res = analyze_media(f)
    pred = [(s["start"], s["end"], s["label"]) for s in res["segments"] if s["label"]!="uncertain"]
    gold = [(s["start"], s["end"], s["label"]) for s in case["segments"]]
    matched_pred=set(); matched_gold=set()
    for i,g in enumerate(gold):
        found = False
        for j,p in enumerate(pred):
            if p[2]==g[2] and iou_1d((p[0],p[1]), (g[0],g[1])) >= 0.5:
                tp += 1; matched_pred.add(j); matched_gold.add(i); found=True; break
        if not found: fn += 1
    for j,p in enumerate(pred):
        if j not in matched_pred: fp += 1

prec = tp / max(tp+fp,1)
rec = tp / max(tp+fn,1)
print(json.dumps({"tp":tp,"fp":fp,"fn":fn,"precision":prec,"recall":rec}, indent=2))
