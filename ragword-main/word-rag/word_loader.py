from docx import Document

class WordLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        doc = Document(self.file_path)
        content = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                content.append(paragraph.text.strip())
        return [{"page_content": " ".join(content), "metadata": {"source": self.file_path}}]
