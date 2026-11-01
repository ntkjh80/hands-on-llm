from pypdf import PdfReader
from pathlib import Path
import json,re,argparse
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description='Extract privately supplied PDF pages and outline for this edition.')
parser.add_argument('pdf',type=Path,help='Path to the user-supplied 598-page PDF')
args=parser.parse_args()
r=PdfReader(args.pdf)
if len(r.pages)!=598: raise SystemExit('This mapping expects the supplied 598-page edition. Check boundaries for another edition.')
(root/'source').mkdir(exist_ok=True)
pages=[page.extract_text() for page in r.pages]
(root/'source/pages.json').write_text(json.dumps(pages,ensure_ascii=False),encoding='utf-8')
outline=[]
def walk(xs,d=0):
 for x in xs:
  if isinstance(x,list):walk(x,d+1)
  else:
   try:outline.append({'title':x.title,'pdf_page':r.get_destination_page_number(x)+1,'depth':d})
   except Exception:pass
walk(r.outline)
starts=[(i,o) for i,o in enumerate(outline) if re.match(r'^\d+\. ',o['title'])]
ko=['언어 모델 입문','토큰과 임베딩','트랜스포머 내부 구조','텍스트 분류','텍스트 군집화와 토픽 모델링','프롬프트 엔지니어링','고급 텍스트 생성과 도구','시맨틱 검색과 RAG','멀티모달 언어 모델','텍스트 임베딩 모델 만들기','표현 모델 미세 조정','생성 모델 미세 조정']
chapters=[]
for n,(idx,o) in enumerate(starts,1):
 end=starts[n][1]['pdf_page']-1 if n<len(starts) else 535
 sections=[]
 for item in outline[idx+1:]:
  if item['depth']<=o['depth']:break
  sections.append(item)
 folder=root/'chapters'/f'ch{n:02d}'
 folder.mkdir(parents=True,exist_ok=True)
 src=root/'source'/f'ch{n:02d}.txt'
 src.write_text('\n\n'.join(f'===== PDF PAGE {p} =====\n{pages[p-1]}' for p in range(o['pdf_page'],end+1)),encoding='utf-8')
 c={'number':n,'title_en':o['title'],'title_ko':ko[n-1],'pdf_start':o['pdf_page'],'pdf_end':end,'source':str(src.relative_to(root)),'directory':str(folder.relative_to(root)),'sections':sections}
 chapters.append(c)
 (folder/'source-map.json').write_text(json.dumps(c,ensure_ascii=False,indent=2))
(root/'source/outline.json').write_text(json.dumps(outline,ensure_ascii=False,indent=2))
(root/'source/chapters.json').write_text(json.dumps(chapters,ensure_ascii=False,indent=2))
figures=[]
for page,text in enumerate(pages,1):
 for kind,label in re.findall(r'(?m)^(Figure|Table)\s+(\d+-\d+)\.[ \t]+\S',text):
  row={'chapter':int(label.split('-')[0]),'label':f'{kind} {label}','pdf_page':page}
  if row not in figures: figures.append(row)
(root/'source/figures.json').write_text(json.dumps(figures,ensure_ascii=False,indent=2))
print(json.dumps([{k:v for k,v in c.items() if k not in ('sections','source','directory')} for c in chapters],ensure_ascii=False,indent=2))
