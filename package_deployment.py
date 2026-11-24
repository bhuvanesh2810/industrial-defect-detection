from pathlib import Path
import zipfile

DEPLOY_DIR = Path("/kaggle/working/industrial_defect_deployment")
ZIP_PATH = Path("/kaggle/working/industrial_defect_deployment.zip")

required = ["app.py", "best.pt", "requirements.txt", "evaluation_metrics.json"]
missing = [name for name in required if not (DEPLOY_DIR / name).is_file()]

if missing:
    raise FileNotFoundError(
        "Deployment folder is missing: " + ", ".join(missing) +
        ". Copy these files into DEPLOY_DIR before creating the ZIP."
    )

optional = ["packages.txt", "runtime.txt", "confusion_matrix.png"]

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as archive:
    for name in required + optional:
        path = DEPLOY_DIR / name
        if path.is_file():
            archive.write(path, arcname=name)

print("Deployment ZIP created:", ZIP_PATH)
print("Included files:", zipfile.ZipFile(ZIP_PATH).namelist())
