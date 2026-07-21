import sys
sys.path.insert(0, '.')
from src.ingest.parse import parse_pdf, _is_likely_heading
from statistics import median
from pathlib import Path

for name in ['oga090_statewide_field_data.pdf', 'uia010_uic_database.pdf', 'wba091_wellbore_database.pdf', 'oda037k_oil_gas_docket.pdf']:
    doc = parse_pdf(Path('data/raw/manual') / name)
    blocks = doc.all_blocks
    sizes = [b.font_size for b in blocks if b.font_size > 0]
    med = median(sizes) if sizes else 0.0
    headings = [b for b in blocks if _is_likely_heading(b.text, b.font_size, med)]
    body = [b for b in blocks if not _is_likely_heading(b.text, b.font_size, med)]
    print(f'--- {name} ---')
    print(f'  total blocks: {len(blocks)}, median font: {med:.1f}')
    print(f'  headings: {len(headings)}, body: {len(body)}')
    print(f'  heading text lengths: min={min((len(h.text) for h in headings), default=0)}, max={max((len(h.text) for h in headings), default=0)}')
    print(f'  first 15 blocks:')
    for b in blocks[:15]:
        flag = 'H' if _is_likely_heading(b.text, b.font_size, med) else 'B'
        print(f'    [{flag}] fs={b.font_size:.1f}  {repr(b.text[:90])}')
    print()
