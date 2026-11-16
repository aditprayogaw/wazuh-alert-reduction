import zipfile
import re
from pathlib import Path

def create_docx(md_path, docx_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    lines = md_text.split('\n')
    
    # Header XML docx
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
        '  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
        '  <Default Extension="xml" ContentType="application/xml"/>\n'
        '  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>\n'
        '</Types>'
    )
    
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        '  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>\n'
        '</Relationships>'
    )
    
    body_elements = []
    
    def escape_xml(s):
        return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&apos;')

    in_table = False
    table_rows = []

    for line in lines:
        line_s = line.strip()
        
        # Table parsing
        if line_s.startswith('|') and line_s.endswith('|'):
            # Check if divider line
            if re.match(r'^\|[\s\-:|]+\|$', line_s):
                continue
            cells = [c.strip() for c in line_s[1:-1].split('|')]
            table_rows.append(cells)
            in_table = True
            continue
        else:
            if in_table and table_rows:
                # Flush table
                tbl_xml = ['<w:tbl><w:tblPr><w:tblW w:w="5000" w:type="pct"/><w:tblBorders><w:top w:val="single" w:sz="4" w:space="0" w:color="auto"/><w:left w:val="single" w:sz="4" w:space="0" w:color="auto"/><w:bottom w:val="single" w:sz="4" w:space="0" w:color="auto"/><w:right w:val="single" w:sz="4" w:space="0" w:color="auto"/><w:insideH w:val="single" w:sz="4" w:space="0" w:color="auto"/><w:insideV w:val="single" w:sz="4" w:space="0" w:color="auto"/></w:tblBorders></w:tblPr>']
                for r_idx, row in enumerate(table_rows):
                    tbl_xml.append('<w:tr>')
                    for cell in row:
                        clean_cell = re.sub(r'[*`]', '', cell)
                        shd = '<w:tcPr><w:shd w:val="clear" w:color="auto" w:fill="EDF2F7"/></w:tcPr>' if r_idx == 0 else ''
                        bold = '<w:b/>' if r_idx == 0 else ''
                        tbl_xml.append(f'<w:tc>{shd}<w:p><w:r><w:rPr>{bold}</w:rPr><w:t>{escape_xml(clean_cell)}</w:t></w:r></w:p></w:tc>')
                    tbl_xml.append('</w:tr>')
                tbl_xml.append('</w:tbl>')
                body_elements.append(''.join(tbl_xml))
                table_rows = []
                in_table = False

        if not line_s or line_s == '---':
            continue

        # Headings
        if line_s.startswith('# '):
            text = escape_xml(line_s[2:].strip())
            body_elements.append(f'<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:rPr><w:b/><w:sz w:val="36"/><w:color w:val="1A365D"/></w:rPr><w:t>{text}</w:t></w:r></w:p>')
        elif line_s.startswith('## '):
            text = escape_xml(line_s[3:].strip())
            body_elements.append(f'<w:p><w:pPr><w:spacing w:before="240" w:after="120"/></w:pPr><w:r><w:rPr><w:b/><w:sz w:val="28"/><w:color w:val="2B6CB0"/></w:rPr><w:t>{text}</w:t></w:r></w:p>')
        elif line_s.startswith('### '):
            text = escape_xml(line_s[4:].strip())
            body_elements.append(f'<w:p><w:pPr><w:spacing w:before="180" w:after="80"/></w:pPr><w:r><w:rPr><w:b/><w:sz w:val="24"/><w:color w:val="2D3748"/></w:rPr><w:t>{text}</w:t></w:r></w:p>')
        elif line_s.startswith('* ') or line_s.startswith('- '):
            raw_text = line_s[2:].strip()
            # Simple bold parsing
            clean_text = re.sub(r'[*`]', '', raw_text)
            body_elements.append(f'<w:p><w:pPr><w:ind w:left="360"/></w:pPr><w:r><w:t>• {escape_xml(clean_text)}</w:t></w:r></w:p>')
        elif re.match(r'^\d+\.\s', line_s):
            raw_text = re.sub(r'^\d+\.\s', '', line_s)
            num = re.match(r'^\d+', line_s).group(0)
            clean_text = re.sub(r'[*`]', '', raw_text)
            body_elements.append(f'<w:p><w:pPr><w:ind w:left="360"/></w:pPr><w:r><w:t>{num}. {escape_xml(clean_text)}</w:t></w:r></w:p>')
        else:
            clean_text = re.sub(r'[*`]', '', line_s)
            body_elements.append(f'<w:p><w:pPr><w:spacing w:after="120"/></w:pPr><w:r><w:t>{escape_xml(clean_text)}</w:t></w:r></w:p>')

    doc_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">\n'
        '  <w:body>\n' +
        '\n'.join(body_elements) +
        '\n  </w:body>\n'
        '</w:document>'
    )

    with zipfile.ZipFile(docx_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', content_types)
        z.writestr('_rels/.rels', rels)
        z.writestr('word/document.xml', doc_xml)

    print(f"File DOCX berhasil dibuat di: {docx_path}")

if __name__ == "__main__":
    create_docx('docs/PANDUAN_PELABELAN_TRIASE.md', 'docs/PANDUAN_PELABELAN_TRIASE.docx')

