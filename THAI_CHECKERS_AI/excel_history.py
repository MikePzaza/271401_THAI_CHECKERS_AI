"""Write a real styled Excel .xlsx history using Python's standard library only.

The small Office Open XML writer avoids extra pip dependencies on Windows.
All text uses inline strings so Thai characters remain readable in Excel.
"""
from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import os

from game_logic import HUMAN, HistoryEntry, move_label, move_numbers

HEADERS = ('ลำดับ', 'ฝ่าย', 'ตาที่ของฝ่าย', 'จากช่อง', 'ไปช่อง',
           'เส้นทางการเดิน', 'ช่องที่กิน', 'จำนวนที่กิน', 'เป็นฮอส', 'หมายเหตุ', 'สถานะ MG400')


def _cell(address: str, value: str | int, style: int = 0) -> str:
    """An inline-text or numeric cell, escaped for valid Office Open XML."""
    if isinstance(value, int):
        return f'<c r="{address}" s="{style}" t="n"><v>{value}</v></c>'
    safe = escape(str(value), quote=False)
    return (f'<c r="{address}" s="{style}" t="inlineStr">'
            f'<is><t xml:space="preserve">{safe}</t></is></c>')


def _row(number: int, values: list[str | int], style: int = 0,
         height: int | None = None) -> str:
    row_attributes = (f' r="{number}" ht="{height}" customHeight="1"'
                      if height is not None else f' r="{number}"')
    return f'<row{row_attributes}>' + ''.join(
        _cell(f'{chr(65 + col)}{number}', value, style)
        for col, value in enumerate(values)
    ) + '</row>'


def _sheet_xml(history: list[HistoryEntry], difficulty: str,
               game_status: str, robot_status: dict[int,str] | None = None) -> str:
    rows = []
    rows.append(_row(1, ['ประวัติการเดินหมากฮอส 8×8 | คน vs ADAMS'], 1, 36))
    rows.append('<row r="2">' + _cell('A2', 'ระดับ AI:', 5) +
                _cell('B2', difficulty, 5) +
                _cell('E2', 'สถานะเกม:', 5) +
                _cell('F2', game_status, 5) + '</row>')
    rows.append(_row(3, ['เลขช่อง 1–64 นับจากซ้ายไปขวา ล่างขึ้นบน (มุมมองคน); คน = น้ำเงิน, ADAMS = ส้ม'], 6))
    rows.append(_row(5, list(HEADERS), 2, 29))
    for index, entry in enumerate(history, start=6):
        path = move_numbers(entry.move)
        details = []
        if entry.captures:
            details.append(f'กิน {len(entry.captures)} ตัว')
        if entry.promoted:
            details.append('เลื่อนเป็นฮอส')
        cells: list[str | int] = [
            entry.ply,
            'คน' if entry.side == HUMAN else 'ADAMS (AI)',
            (entry.ply + 1) // 2,
            path[0],
            path[-1],
            move_label(entry.move),
            ', '.join(map(str, entry.captures)) if entry.captures else '—',
            len(entry.captures),
            'ใช่' if entry.promoted else 'ไม่',
            ' | '.join(details) if details else 'เดินปกติ',
            ('MANUAL / HUMAN' if entry.side == HUMAN else (robot_status or {}).get(entry.ply, 'NO ROBOT DATA')),
        ]
        rows.append(_row(index, cells, 3 if entry.side == HUMAN else 4, 25))

    last_row = max(5, len(history) + 5)
    widths = (11, 17, 15, 13, 13, 25, 22, 16, 15, 25, 32)
    columns = ''.join(
        f'<col min="{i}" max="{i}" width="{w}" customWidth="1"/>'
        for i, w in enumerate(widths, 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<dimension ref="A1:K' + str(last_row) + '"/>'
        '<sheetViews><sheetView workbookViewId="0">'
        '<pane ySplit="5" topLeftCell="A6" activePane="bottomLeft" state="frozen"/>'
        '</sheetView></sheetViews>'
        '<sheetFormatPr defaultRowHeight="20"/>'
        '<cols>' + columns + '</cols>'
        '<sheetData>' + ''.join(rows) + '</sheetData>'
        '<mergeCells count="2"><mergeCell ref="A1:K1"/>'
        '<mergeCell ref="A3:K3"/></mergeCells>'
        f'<autoFilter ref="A5:K{last_row}"/>'
        '<pageMargins left="0.3" right="0.3" top="0.5" bottom="0.5" header="0.3" footer="0.3"/>'
        '</worksheet>'
    )


_CONTENT_TYPES = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
</Types>'''

_ROOT_RELS = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
</Relationships>'''

_WORKBOOK = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets><sheet name="ประวัติการเดิน" sheetId="1" r:id="rId1"/></sheets>
</workbook>'''

_WORKBOOK_RELS = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>'''

# Six styles: default, navy title, blue column headings, pale-blue human,
# pale-orange AI, bold metadata, subdued explanation.
_STYLES = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<fonts count="4">
<font><sz val="11"/><name val="Tahoma"/><color rgb="FF18304A"/></font>
<font><b/><sz val="16"/><name val="Tahoma"/><color rgb="FFFFFFFF"/></font>
<font><b/><sz val="11"/><name val="Tahoma"/><color rgb="FFFFFFFF"/></font>
<font><b/><sz val="11"/><name val="Tahoma"/><color rgb="FF18304A"/></font>
</fonts>
<fills count="6">
<fill><patternFill patternType="none"/></fill>
<fill><patternFill patternType="gray125"/></fill>
<fill><patternFill patternType="solid"><fgColor rgb="FF18304A"/><bgColor indexed="64"/></patternFill></fill>
<fill><patternFill patternType="solid"><fgColor rgb="FF2876CF"/><bgColor indexed="64"/></patternFill></fill>
<fill><patternFill patternType="solid"><fgColor rgb="FFE7F3FF"/><bgColor indexed="64"/></patternFill></fill>
<fill><patternFill patternType="solid"><fgColor rgb="FFFFECE5"/><bgColor indexed="64"/></patternFill></fill>
</fills>
<borders count="2"><border><left/><right/><top/><bottom/><diagonal/></border>
<border><left/><right/><top/><bottom style="hair"><color rgb="FFDDE4EB"/></bottom><diagonal/></border></borders>
<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
<cellXfs count="7">
<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"><alignment vertical="center" indent="1"/></xf>
<xf numFmtId="0" fontId="2" fillId="3" borderId="0" xfId="0" applyFont="1" applyFill="1"><alignment vertical="center" horizontal="center"/></xf>
<xf numFmtId="0" fontId="0" fillId="4" borderId="1" xfId="0" applyFill="1" applyBorder="1"><alignment vertical="center" horizontal="center"/></xf>
<xf numFmtId="0" fontId="0" fillId="5" borderId="1" xfId="0" applyFill="1" applyBorder="1"><alignment vertical="center" horizontal="center"/></xf>
<xf numFmtId="0" fontId="3" fillId="0" borderId="0" xfId="0" applyFont="1"><alignment vertical="center"/></xf>
<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"><alignment vertical="center"/></xf>
</cellXfs>
<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>'''


def export_history_xlsx(path: str | Path, history: list[HistoryEntry], *,
                        difficulty: str, game_status: str,
                        robot_status: dict[int,str] | None = None) -> Path:
    """Export every ply in order. Suitable for opening directly with Excel."""
    target = Path(path)
    if target.suffix.lower() != '.xlsx':
        raise ValueError('ชื่อไฟล์ต้องลงท้ายด้วย .xlsx')
    temp = target.with_name(target.name + '.tmp')
    created = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    core = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"'
        ' xmlns:dc="http://purl.org/dc/elements/1.1/"'
        ' xmlns:dcterms="http://purl.org/dc/terms/"'
        ' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        '<dc:creator>Thai Checkers ADAMS Desktop</dc:creator>'
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{created}</dcterms:created>'
        '</cp:coreProperties>'
    )
    try:
        with ZipFile(temp, 'w', ZIP_DEFLATED) as z:
            z.writestr('[Content_Types].xml', _CONTENT_TYPES)
            z.writestr('_rels/.rels', _ROOT_RELS)
            z.writestr('xl/workbook.xml', _WORKBOOK)
            z.writestr('xl/_rels/workbook.xml.rels', _WORKBOOK_RELS)
            z.writestr('xl/styles.xml', _STYLES)
            z.writestr('xl/worksheets/sheet1.xml', _sheet_xml(history, difficulty, game_status, robot_status))
            z.writestr('docProps/core.xml', core)
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()
    return target
