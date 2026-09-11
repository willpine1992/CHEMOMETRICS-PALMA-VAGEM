"""
Baixa a planilha fonte do Google Drive (live) a cada execucao do ETL, em vez de
usar uma copia estatica local. Isso garante que o banco SQL sempre reflita a
versao mais atual editada pelo usuario no Drive.
"""
import os

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
CREDS_PATH = "/Users/macbookpro/Documents/POSDOC - MAC/QUIMIOINFORMÁTICA/chemoinformatics-proj-71936-629b8d6ee88a.json"
FILE_ID = "1wT33uLHM6rwS1SpiXJ6c9ypj5SaI2X4c"
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "isotermas_vagem_drive.xlsx")


def fetch_xlsx(force=True):
    """Baixa o xlsx do Drive para CACHE_PATH e retorna o caminho local. force=True sempre rebaixa."""
    if not force and os.path.exists(CACHE_PATH):
        return CACHE_PATH

    creds = service_account.Credentials.from_service_account_file(CREDS_PATH, scopes=SCOPES)
    service = build("drive", "v3", credentials=creds)

    meta = service.files().get(fileId=FILE_ID, fields="id,name,mimeType,modifiedTime,size").execute()
    mime = meta.get("mimeType")

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

    if mime == "application/vnd.google-apps.spreadsheet":
        # Planilha Google nativa -> precisa exportar para xlsx
        request = service.files().export_media(
            fileId=FILE_ID,
            mimeType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        # Binario .xlsx real -> download direto, sem export
        request = service.files().get_media(fileId=FILE_ID)

    with open(CACHE_PATH, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _status, done = downloader.next_chunk()

    size = os.path.getsize(CACHE_PATH)
    print(f"[drive_source] Baixado do Drive: name={meta.get('name')!r} mimeType={mime} "
          f"modifiedTime={meta.get('modifiedTime')} drive_size={meta.get('size')} "
          f"local_bytes={size} -> {CACHE_PATH}")
    return CACHE_PATH


if __name__ == "__main__":
    path = fetch_xlsx(force=True)
    print("OK:", path)
