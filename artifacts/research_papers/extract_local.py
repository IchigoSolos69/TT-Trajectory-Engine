from pathlib import Path
import hashlib, json
import pdfplumber

root = Path(__file__).resolve().parent
paths = [
    Path(r'C:\Users\Adi\Downloads\2509.18387v1.pdf'),
    Path(r'C:\Users\Adi\Downloads\video-based-table-tennis-tracking-and-trajectory-prediction-using-convolutional-neural-networks.pdf'),
]
manifest = []
for path in paths:
    with pdfplumber.open(path) as doc:
        pages = [page.extract_text() or '' for page in doc.pages]
        destination = root / (path.stem + '.txt')
        destination.write_text('\n\n'.join(f'PAGE {i+1}\n{text}' for i,text in enumerate(pages)), encoding='utf-8')
        manifest.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'pages':len(pages),'metadata':doc.metadata,'text':str(destination)})
        print(json.dumps({'file':path.name,'pages':len(pages),'first_page':pages[0]},ensure_ascii=False))
        for index in range(min(2,len(pages))):
            doc.pages[index].to_image(resolution=100).save(root / f'{path.stem}_page_{index+1}.png')
root.joinpath('local_pdf_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
