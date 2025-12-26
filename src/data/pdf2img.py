"""
PDF to Image Converter Module

PDF 파일 또는 디렉토리를 받아서 각 페이지를 이미지로 변환합니다.
파일명 형식: {pdf명}_p{page:04d}.{format}
"""

from pathlib import Path

from tqdm import tqdm

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


class PDF2ImageConverter:
    """PDF를 페이지별 이미지로 변환합니다."""

    def __init__(
        self,
        dpi: int = 200,
        image_format: str = "png",
    ):
        if fitz is None:
            raise ImportError(
                "PyMuPDF is required for PDF processing. "
                "Install it with: pip install pymupdf"
            )
        self.dpi = dpi
        self.image_format = image_format.lower()
        self.zoom = dpi / 72  # PDF default is 72 DPI

    def convert_single_pdf(
        self,
        pdf_path: str | Path,
        output_dir: str | Path,
        start_page: int | None = None,
        end_page: int | None = None,
    ) -> list[Path]:
        """
        단일 PDF를 페이지별 이미지로 변환합니다.

        Args:
            pdf_path: PDF 파일 경로
            output_dir: 출력 이미지 디렉토리
            start_page: 시작 페이지 (1-indexed, None이면 처음부터)
            end_page: 끝 페이지 (1-indexed, None이면 끝까지)

        Returns:
            생성된 이미지 파일 경로 리스트
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        # 1-indexed to 0-indexed 변환
        start = (start_page - 1) if start_page is not None else 0
        end = end_page if end_page is not None else total_pages

        # 범위 검증
        start = max(0, start)
        end = min(end, total_pages)

        image_paths = []
        pdf_name = pdf_path.stem

        for page_num in tqdm(
            range(start, end),
            desc=f"Converting {pdf_name}",
            leave=False,
        ):
            page = doc[page_num]
            mat = fitz.Matrix(self.zoom, self.zoom)
            pix = page.get_pixmap(matrix=mat)

            # 파일명: {pdf명}_p{page:04d}.{format} (1-indexed)
            image_filename = f"{pdf_name}_p{page_num + 1:04d}.{self.image_format}"
            image_path = output_dir / image_filename
            pix.save(str(image_path))
            image_paths.append(image_path)

        doc.close()
        return image_paths

    def convert_directory(
        self,
        pdf_dir: str | Path,
        output_dir: str | Path,
    ) -> list[Path]:
        """
        디렉토리 내 모든 PDF를 이미지로 변환합니다.

        Args:
            pdf_dir: PDF 파일들이 있는 디렉토리
            output_dir: 출력 이미지 디렉토리

        Returns:
            생성된 이미지 파일 경로 리스트
        """
        pdf_dir = Path(pdf_dir)
        output_dir = Path(output_dir)

        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in {pdf_dir}")

        all_image_paths = []

        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            image_paths = self.convert_single_pdf(pdf_path, output_dir)
            all_image_paths.extend(image_paths)

        return all_image_paths

    def convert(
        self,
        source: str | Path,
        output_dir: str | Path,
        start_page: int | None = None,
        end_page: int | None = None,
    ) -> list[Path]:
        """
        PDF 파일 또는 디렉토리를 이미지로 변환합니다.

        Args:
            source: PDF 파일 경로 또는 PDF가 있는 디렉토리 경로
            output_dir: 출력 이미지 디렉토리
            start_page: 시작 페이지 (단일 PDF인 경우, 1-indexed)
            end_page: 끝 페이지 (단일 PDF인 경우, 1-indexed)

        Returns:
            생성된 이미지 파일 경로 리스트
        """
        source = Path(source)
        output_dir = Path(output_dir)

        if source.is_file() and source.suffix.lower() == ".pdf":
            return self.convert_single_pdf(source, output_dir, start_page, end_page)
        elif source.is_dir():
            return self.convert_directory(source, output_dir)
        else:
            raise ValueError(
                f"Source must be a PDF file or directory containing PDFs: {source}"
            )


def pdf2img(
    source: str | Path,
    output_dir: str | Path,
    dpi: int = 200,
    image_format: str = "png",
    start_page: int | None = None,
    end_page: int | None = None,
) -> list[Path]:
    """
    PDF를 이미지로 변환하는 편의 함수.

    Args:
        source: PDF 파일 경로 또는 PDF가 있는 디렉토리 경로
        output_dir: 출력 이미지 디렉토리
        dpi: 이미지 해상도 (기본: 200)
        image_format: 이미지 포맷 (기본: png)
        start_page: 시작 페이지 (단일 PDF인 경우, 1-indexed)
        end_page: 끝 페이지 (단일 PDF인 경우, 1-indexed)

    Returns:
        생성된 이미지 파일 경로 리스트

    Example:
        >>> from src.data.pdf2img import pdf2img
        >>> images = pdf2img("document.pdf", "./images", dpi=300)
        >>> print(images)
        [PosixPath('images/document_p0001.png'), PosixPath('images/document_p0002.png'), ...]
    """
    converter = PDF2ImageConverter(dpi=dpi, image_format=image_format)
    return converter.convert(source, output_dir, start_page, end_page)
