import io
import hashlib
from pypdf import PdfReader
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document


class PDFLoader:
    def __init__(self, stream: bytes):
        self.stream = stream
        self.reader = PdfReader(io.BytesIO(stream))
        self.docId  = hashlib.md5(stream).hexdigest()

    def load(self):
        return [
            Document(
                page_content=page.extract_text() or "",
                metadata={"docId": self.docId, "page": idx}
            )
            for idx, page in enumerate(self.reader.pages)
        ]