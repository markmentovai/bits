#!/usr/bin/env python3

# Copyright 2023 Mark Mentovai
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License along
# with this program. If not, see <https://www.gnu.org/licenses/>.

#
"""
Extract arrival alert notices from chart supplement PDFs.

The FAA Runway Safety Program’s Arrival Alert Notice is described at
https://www.faa.gov/airports/runway_safety/hotspots/aan, and was discussed in
the FAA Aeronautical Charting Meeting, Charting Group, under RD 20-02-345.
https://www.faa.gov/air_traffic/flight_info/aeronav/acf/media/RDs/20-02-345_Wrong_Surface_Hot_Spots.pdf

This program extracts arrival alert notices as PDFs from FAA chart supplement
PDFs, which may be obtained from
https://www.faa.gov/air_traffic/flight_info/aeronav/digital_products/dafd/. It
understands both full chart supplement volumes (listed by FAA as “cover-to-cover
PDFs” and normally named CS_VOL_YYYYMMDD.pdf) and “rear matter” PDFs (listed as
“application data”, offered as a downloadable suite under the DCS_YYYYMMDD.zip
name, with individual rear matter files normally named VOL_rear_DDMMYYYY.pdf).
It is also able to operate directly on .zip files containing supported PDFs, or
directories expanded from such .zip files. Through the use of the requests
module, URLs to these files can also be provided to this program directly. In
this case, the referenced file will be downloaded (but not saved) from the URL
and processed.

Output files are placed in the same directory as the input, or can be saved in a
different location by using the --out-dir option. Output files are named
identically to the input (PDF, .zip, or directory) that they were extracted
from, but with “_aan_” and their FAA airport identifier added to the filename.
Examples: CS_NE_20231005_aan_ROC.pdf, PAC_rear_05OCT2023_aan_HNL.pdf. At some
airports, more than one arrival alert notice page is published in the Chart
Supplement, and for these airports, multiple pages will appear in their output
PDFs.

Optionally, machine-readable CSV output can be produced identifying arrival
alert notices and their source documents (--csv). This feature may be used with
extraction disabled (--no-pdf-output) as a means of identifying the notices for
the benefit of another tool capable of operating directly with pages within PDF
documents.

When producing CSV output, text may be extracted from arrival alert notices
(option --text). Some notices do not contain machine-readable text. In these
cases, OCR can be used to recover the text from such notices (option
--text-ocr). Tesseract is used to perform OCR; both Tesseract and
English-language data (tessdata-eng) must be installed for this feature to work,
and it may also be necessary to set the TESSDATA_PREFIX environment variable to
the location of Tesseract data (such as /usr/share/tessdata). OCR will only be
used to recover text that is not machine-readable.

The chart supplement header and footer may optionally be removed from extracted
arrival alert notices (--no-header-footer).

This program depends on pymupdf (module fitz) for PDF manipulation.

This has been tested with chart supplements from cycle 2304 (2023-04-20) and
2310 (2023-10-05).
"""

import argparse
import collections
import csv
import datetime
import io
import os
import re
import sys
import zipfile

import fitz
import requests


class FaaCsAanException(Exception):
    pass


def _first_non_none(*sequence):
    """Returns the first item in `sequence` that isn’t None, or None if there
    are none.
    """
    for item in sequence:
        if item is not None:
            return item
    return None


class _FitzDocument_SaveOnClose(fitz.Document):
    """A wrapper for fitz.Document that saves a file when closed."""

    def __init__(self, filename=None, stream=None, *, new=False, **kwargs):
        self._filename = filename
        self._incremental = not new

        if new and not stream:
            super().__init__(None, stream, **kwargs)
        else:
            super().__init__(filename, stream, **kwargs)

    def close(self):
        super().ez_save(self._filename, incremental=self._incremental)


class _FitzDocument_WithDisplayPath(fitz.Document):
    """A wrapper for fitz.Document that carries an additional `display_path`
    property.

    The `display_path` property can be useful to identify the provenance of PDFs
    extracted from .zip files.
    """

    def __init__(self,
                 filename=None,
                 stream=None,
                 *,
                 display_path=None,
                 local=True,
                 **kwargs):
        self._display_path = display_path
        self._local = local
        super().__init__(filename, stream, **kwargs)

    @property
    def display_path(self):
        if self._display_path is not None:
            return self._display_path
        else:
            return self.name

    @property
    def local(self):
        return self._local


class _PdfSource:
    """Provider of a single PDF, or a .zip or directory containing PDFs."""

    class _Pdf:

        def __init__(self, path, *, data=None, local=True):
            if data is None:
                self._pdf = _FitzDocument_WithDisplayPath(path, local=local)
            else:
                self._pdf = _FitzDocument_WithDisplayPath(stream=data,
                                                          display_path=path,
                                                          local=local)

        def pdfs(self):
            yield self._pdf

    class _Zip:

        def __init__(self, path, member_re, *, data=None, local=True):
            if data is None:
                self._zip = zipfile.ZipFile(path)
            else:
                self._zip = zipfile.ZipFile(io.BytesIO(data))

            self._member_re = member_re
            self._display_path_base = path
            self._local = local

        def pdfs(self):
            for member_zipinfo in self._zip.infolist():
                if os.sep in member_zipinfo.filename:
                    # Only process members at the .zip file’s root, for
                    # compatibility with _Dir.
                    continue
                if (self._member_re is not None and
                        not self._member_re.match(member_zipinfo.filename)):
                    continue

                yield _FitzDocument_WithDisplayPath(
                    stream=self._zip.read(member_zipinfo),
                    display_path=os.path.join(self._display_path_base,
                                              member_zipinfo.filename),
                    local=self._local)

    class _Dir:

        def __init__(self, path, member_re):
            self._path = path
            self._member_re = member_re

        def pdfs(self):
            for name in os.listdir(self._path):
                if (self._member_re is not None and
                        not self._member_re.match(name)):
                    continue
                yield _FitzDocument_WithDisplayPath(
                    os.path.join(self._path, name))

    def __init__(self, path, member_re=None, session=None):
        # Don’t try self._Pdf first, because fitz.Document doesn’t raise an
        # immediate exception when given an invalid PDF.

        try:
            if session is None:
                session = requests

            # This will only work if `path` is a URL starting with a scheme, and
            # the scheme is something that `session` is able to handle.
            # Normally, this will only be http:// and https://.
            response = session.get(path)
            response.raise_for_status()
            try:
                self._source = self._Zip(path,
                                         member_re,
                                         data=response.content,
                                         local=False)
            except zipfile.BadZipFile:
                self._source = self._Pdf(path,
                                         data=response.content,
                                         local=False)
        except (requests.exceptions.MissingSchema,
                requests.exceptions.InvalidSchema):
            # In the more common case, `path` is a filesystem pathname.
            try:
                self._source = self._Zip(path, member_re)
            except IsADirectoryError:
                self._source = self._Dir(path, member_re)
            except zipfile.BadZipFile:
                self._source = self._Pdf(path)

    def pdfs(self):
        yield from self._source.pdfs()


_aan_tuple = collections.namedtuple(
    '_aan_tuple', ('faa_id', 'airport_name', 'cs_pdf_page', 'page_label'))

_PAGE_HEADER_RE = re.compile(r'^(?:(?P<even_page_label>[A-Z]?[0-9]+)(?:\n|$))?'
                             r'(?P<section_name>.*)'
                             r'(?:\n(?P<odd_page_label>[A-Z]?[0-9]+))?$')

_ARRIVAL_ALERT_RE = re.compile(r'^(?P<airport_name>.+) '
                               r'\((?P<faa_id>[A-Z0-9]{3,4})\) '
                               r'ARRIVAL ALERT$')


def cs_arrival_alert_notices(cs_pdf):
    """Processes a chart supplement PDF, yielding an _aan_tuple for each Arrival
    Alert Notice page contained therein."""

    _TOC_SPECIAL_NOTICES_RE = re.compile(
        r'^Special Notices *\.+ *(?P<page_label>[0-9]+)$')

    found_toc = False
    found_special_notices = False
    page_number = -1
    while page_number < cs_pdf.page_count - 1:
        page_number += 1
        page = cs_pdf.load_page(page_number)

        text_blocks = page.get_text('blocks',
                                    flags=fitz.TEXT_PRESERVE_LIGATURES |
                                    fitz.TEXT_PRESERVE_WHITESPACE |
                                    fitz.TEXT_MEDIABOX_CLIP,
                                    sort=True)
        if len(text_blocks) < 2:
            continue

        page_header_text = text_blocks[0][4].rstrip('\n')

        if (page_header_text
                in ('UNITED STATES GOVERNMENT FLIGHT INFORMATION PUBLICATION',
                    'GENERAL INFORMATION', 'PIREP FORM')):
            # This is the cover or title page of a complete Chart Supplement
            # volume, or the last unnumbered page before the rear cover.
            continue

        if (page_header_text.startswith('I. POSITION REPORTS\n')):
            # This is the back cover of the complete AK or PAC volume.
            continue

        page_header_match = _PAGE_HEADER_RE.match(page_header_text)
        page_label = _first_non_none(page_header_match.group('even_page_label'),
                                     page_header_match.group('odd_page_label'))
        section_name = page_header_match.group('section_name')

        if (not found_toc and section_name == 'GENERAL INFORMATION' and
                text_blocks[1][4].rstrip('\n') == 'TABLE OF CONTENTS'):
            # If processing a full CS volume, read the table of contents to
            # determine where the SPECIAL NOTICES section begins, to be able to
            # skip directly to it.
            found_toc = True
            for toc_text_block in text_blocks[2:]:
                toc_text_block = toc_text_block[4].rstrip('\n')
                for toc_line in toc_text_block.split('\n'):
                    toc_special_notices_match = (
                        _TOC_SPECIAL_NOTICES_RE.match(toc_line))
                    if not toc_special_notices_match:
                        continue

                    page_offset = int(
                        toc_special_notices_match.group('page_label')) - int(
                            page_label)
                    if page_offset <= 0:
                        raise FaaCsAanException(
                            'TOC indicates notices precede TOC')

                    # Subtract 1 to counteract the `page_number += 1` at the top
                    # of the loop body.
                    page_number += page_offset - 1

        if section_name not in ('SPECIAL NOTICES', 'NOTICES'):
            # Only process the SPECIAL NOTICES section. In the AK volume,
            # special notices use the NOTICES header.
            if found_special_notices:
                # If notices were already found and the header’s section name
                # has changed, there must be no more notices to process.
                break
            continue

        found_special_notices = True

        # Replace newlines within the first text block with spaces, to cover
        # cases such as KRHV, where the arrival alert notice title is split
        # between two lines.
        first_line_text = re.sub(r'[ \n]+', ' ', text_blocks[1][4].rstrip('\n'))
        arrival_alert_match = _ARRIVAL_ALERT_RE.match(first_line_text)
        if not arrival_alert_match:
            # Only process arrival alert notices.
            continue

        # This is an arrival alert notice.
        yield _aan_tuple(arrival_alert_match.group('faa_id'),
                         arrival_alert_match.group('airport_name'), page,
                         page_label)

    if not found_special_notices:
        raise FaaCsAanException('did not find SPECIAL NOTICES section')


def _validate_cs_aan_pdf_text_blocks(blocks):
    # text_blocks[0] will be the header line, and text_blocks[-1] will be the
    # footer line. There may be a horizontal rule between the header line and
    # the page content, and there may be a horizontal rule between the page
    # content and the footer line, so remove things above text_blocks[1] and
    # below text_blocks[-2]. That means that there must be at least three text
    # blocks: a header, at least some content, and a footer.
    if len(blocks) < 3:
        raise FaaCsAanException('too few text blocks')

    # Check the header, the beginning of the content, and the footer against
    # expected patterns to be safe, because removal is a destructive operation.
    if not _PAGE_HEADER_RE.match(blocks[0][4]):
        raise FaaCsAanException('unexpected header text')

    if not _ARRIVAL_ALERT_RE.match(
            re.sub(r'[ \n]+', ' ', blocks[1][4].rstrip('\n'))):
        raise FaaCsAanException('unexpected title text')

    # “VOL, EFFECTIVE_DATE to EXPIRY_DATE”. Example: “NE, 5 OCT 2023 to 30 NOV
    # 2023”.
    if not re.match(
            r'^[A-Z]{2,3}, '
            r'[0-9]{1,2} [A-Z]{3} [0-9]{4} to '
            r'[0-9]{1,2} [A-Z]{3} [0-9]{4}$', blocks[-1][4]):
        raise FaaCsAanException('unexpected footer text')


def _remove_header_footer(aan_pdf_page):
    text_blocks = aan_pdf_page.get_text('blocks',
                                        flags=fitz.TEXT_PRESERVE_LIGATURES |
                                        fitz.TEXT_PRESERVE_WHITESPACE |
                                        fitz.TEXT_MEDIABOX_CLIP,
                                        sort=True)

    _validate_cs_aan_pdf_text_blocks(text_blocks)

    aan_pdf_page.add_redact_annot(fitz.Rect(aan_pdf_page.mediabox.x0,
                                            aan_pdf_page.mediabox.y0,
                                            aan_pdf_page.mediabox.x1,
                                            text_blocks[1][1]),
                                  fill=(1, 1, 1))
    aan_pdf_page.add_redact_annot(fitz.Rect(aan_pdf_page.mediabox.x0,
                                            text_blocks[-2][3],
                                            aan_pdf_page.mediabox.x1,
                                            aan_pdf_page.mediabox.y1),
                                  fill=(1, 1, 1))
    aan_pdf_page.apply_redactions()


def extract_cs_arrival_alert_notices(cs_pdf,
                                     cs_path,
                                     *,
                                     output_pdf=True,
                                     out_dir=None,
                                     remove_header_footer=False,
                                     callback=None):
    """Extracts arrival alert notice PDFs from a chart supplement PDF."""

    # Remove trailing separators from cs_path, so that when operating on a
    # directory, trailing separators don’t cause airport_aan_path to be computed
    # improperly.
    cs_path = cs_path.rstrip(os.sep +
                             (os.altsep if os.altsep is not None else ''))

    last_aan_faa_id = None
    aan_pdf = None
    try:
        for aan in cs_arrival_alert_notices(cs_pdf):
            if output_pdf and last_aan_faa_id != aan.faa_id:
                last_aan_faa_id = aan.faa_id

                if aan_pdf is not None:
                    aan_pdf.close()

                airport_aan_path = re.sub(r'(\.(pdf|zip))?$',
                                          '_aan_' + aan.faa_id + '.pdf',
                                          cs_path, 1, re.IGNORECASE)
                if not cs_pdf.local:
                    airport_aan_path = os.path.basename(airport_aan_path)
                if out_dir is not None:
                    airport_aan_path = os.path.join(
                        out_dir, os.path.basename(airport_aan_path))

                aan_pdf = _FitzDocument_SaveOnClose(airport_aan_path, new=True)

            if callback is not None:
                callback(aan, airport_aan_path if output_pdf else None,
                         aan_pdf.page_count if output_pdf else None)

            if aan_pdf is not None:
                aan_pdf.insert_pdf(aan.cs_pdf_page.parent,
                                   aan.cs_pdf_page.number,
                                   aan.cs_pdf_page.number)
                if remove_header_footer:
                    aan_pdf_page = aan_pdf.load_page(aan_pdf.page_count - 1)
                    _remove_header_footer(aan_pdf_page)
    finally:
        if aan_pdf is not None:
            aan_pdf.close()


def _string_to_date(s):
    # Don’t use datetime.datetime.strptime because it’s locale-sensitive.
    # Attempt to match both short (abbreviated) and long (full) month names.
    # Most use the abbreviated form (“19 MAY 2022”) but VGT uses the full form
    # (“29 DECEMBER 2022”).
    _MONTHS = (
        'January',
        'February',
        'March',
        'April',
        'May',
        'June',
        'July',
        'August',
        'September',
        'October',
        'November',
        'December',
    )
    _LONG_MONTH_TO_NUM = dict(
        (enum[1].lower(), enum[0]) for enum in enumerate(_MONTHS, 1))
    _SHORT_MONTH_TO_NUM = dict(
        (enum[1].lower()[:3], enum[0]) for enum in enumerate(_MONTHS, 1))

    match = re.match(r'([0-9]{1,2}) ([A-Z]{3,}) ([0-9]{4})$', s, re.IGNORECASE)
    return datetime.date(
        int(match.group(3)),
        _first_non_none(
            _SHORT_MONTH_TO_NUM.get(match.group(2)[:3].lower(), None),
            _LONG_MONTH_TO_NUM.get(match.group(2).lower(), None)),
        int(match.group(1)))


_text_tuple = collections.namedtuple('_text_tuple', (
    'landing_direction',
    'locations',
    'title',
    'text',
    'disclaimers',
    'email',
    'effective_from',
    'effective_to',
    'used_ocr',
))


def extract_text_from_cs_arrival_alert_notice(aan_pdf_page, *, allow_ocr=False):
    blocks = aan_pdf_page.get_text(
        'blocks',
        flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE |
        fitz.TEXT_MEDIABOX_CLIP | fitz.TEXT_PRESERVE_IMAGES,
        sort=True)
    _validate_cs_aan_pdf_text_blocks(blocks)
    blocks = blocks[1:-1]

    image_rects = [fitz.Rect(block[0:4]) for block in blocks if block[6] == 1]

    # Text blocks are blocks that are not images and that do not overlap any
    # image. This filters out text that overlays and forms a composite with an
    # image.
    text_blocks = [
        block for block in blocks if block[6] == 0 and not any(
            fitz.Rect(block[0:4]).intersects(image_rect)
            for image_rect in image_rects)
    ]

    # If any Unicode replacement characters (u+fffd) are found and OCR is
    # allowed, try to use OCR to read the text.
    used_ocr = False
    use_blocks = text_blocks
    if (allow_ocr and
            any('\ufffd' in text_block[4] for text_block in text_blocks)):
        # fitz.get_tessdata is new in 1.23.
        textpage_ocr = aan_pdf_page.get_textpage_ocr(
            flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE |
            fitz.TEXT_MEDIABOX_CLIP,
            dpi=300,
            full=True,
            tessdata=(fitz.get_tessdata()
                      if hasattr(fitz, 'get_tessdata') else None))
        ocr_blocks = textpage_ocr.extractBLOCKS()
        ocr_blocks = [
            ocr_block for ocr_block in ocr_blocks if not any(
                fitz.Rect(ocr_block[0:4]).intersects(image_rect)
                for image_rect in image_rects)
        ]

        use_blocks = []
        for text_block in text_blocks:
            if '\ufffd' in text_block[4]:
                text_block_rect = fitz.Rect(text_block[0:4])
                replacement_ocr_blocks = [
                    ocr_block for ocr_block in ocr_blocks
                    if fitz.Rect(ocr_block[0:4]).intersects(text_block_rect)
                ]
                for ocr_block in replacement_ocr_blocks:
                    used_ocr = True
                    # Consume, because a single ocr_block may intersect with
                    # multiple text_blocks.
                    ocr_blocks.remove(ocr_block)

                    use_blocks.append(ocr_block)
            else:
                use_blocks.append(text_block)

    text = '\n'.join(block[4].rstrip(' \n') for block in use_blocks[1:])

    landing_direction_match = re.match(
        r'Landing ((?:(?:North|South)(?:east|west)?)|East|West)(?=[^a-z]|$)',
        text, re.IGNORECASE)
    landing_direction = {
        'North': 'N',
        'Northeast': 'NE',
        'East': 'E',
        'Southeast': 'SE',
        'South': 'S',
        'Southwest': 'SW',
        'West': 'W',
        'Northwest': 'NW',
    }[landing_direction_match.group(1)[0].upper() +
      landing_direction_match.group(1)[1:].lower()]
    text = (text[:landing_direction_match.start()] +
            text[landing_direction_match.end():]).lstrip(' \n')

    locations = []
    # At PSP landing SE, locations are given as “RWY 13L and RWY 13 R and TWY
    # C”. At PSP landing NW, locations are given as “RWY31L and RWY 31R and TWY
    # C”. Take care to match the forms with odd spacing (“RWY 13 R”, “RWY31L”).
    location_pattern = (r'((?:(?i:RWY) ?[0-9]{1,2}(?: ?[LCR])?)|'
                        r'(?:(?i:TWY) [A-Z]{1,2}[0-9]{0,2}))')
    location_conjunction = ''
    while True:
        location_match = re.match(location_conjunction + location_pattern, text)
        if location_match is None:
            break

        location = location_match.group(1).upper()

        # Standardize forms with odd spacing (described above) to expected forms
        # (“RWY 13R”, “RWY 31L”).
        location = re.sub(r'^(RWY)([0-9]+)', r'\1 \2', location)
        location = re.sub(r'(RWY [0-9]+) ([LCR])', r'\1\2', location)

        locations.append(location)
        text = (text[:location_match.start()] +
                text[location_match.end():]).lstrip(' \n')

        if location_conjunction == '':
            location_conjunction = r'(?:(?:,|, and|and) )'

    effectivity_match = re.search(
        r'Effective ([0-9]{1,2} [A-Z]{3,} [0-9]{4}) '
        r'to ([0-9]{1,2} [A-Z]{3,} [0-9]{4})$', text)
    if effectivity_match is not None:
        effective_from = _string_to_date(effectivity_match.group(1))
        effective_to = _string_to_date(effectivity_match.group(2))
        text = text[:effectivity_match.start()].rstrip(' \n')
    else:
        effective_from = None
        effective_to = None

    email_match = re.search('For Inquiries: ([0-9a-z._-]+@[0-9a-z.-]+)\.?$',
                            text, re.IGNORECASE)
    if email_match is not None:
        email = email_match.group(1)
        text = text[:email_match.start()].rstrip(' \n')
    else:
        email = None

    disclaimers_match = re.search(
        'Not for navigation|For situational awareness', text, re.IGNORECASE)
    disclaimers = text[disclaimers_match.start():].split('\n')
    text = text[:disclaimers_match.start()].rstrip(' \n')

    title_match = re.match(r'((?!Pilot)[^\n]*)(?:\n|$)', text, re.IGNORECASE)
    if title_match is not None:
        title = title_match.group(1).rstrip(' \n')
        text = text[title_match.end():].lstrip(' \n')
    else:
        title = None

    text = text.replace('\n', ' ')
    text = re.sub(r' {2,}', ' ', text)

    return _text_tuple(
        landing_direction,
        locations,
        title,
        text,
        disclaimers,
        email,
        effective_from,
        effective_to,
        used_ocr,
    )


class _Callback:

    def __init__(self,
                 *,
                 quiet=False,
                 use_csv=False,
                 text=False,
                 text_ocr=False,
                 pdf_output=True):
        self._quiet = quiet
        self._csv = (csv.writer(sys.stdout, lineterminator=os.linesep)
                     if use_csv else None)
        self._text = text
        self._text_ocr = text_ocr
        self._pdf_output = pdf_output
        self._needs_csv_header = True
        self._last_airport_aan_path = None

    def _callback(self, aan, airport_aan_path, airport_aan_page):
        if self._quiet:
            return

        if self._csv is not None:
            if self._needs_csv_header:
                self._needs_csv_header = False

                row = [
                    'cs_pdf_path',
                    'cs_pdf_page_number',
                    'cs_pdf_page_label',
                    'faa_id',
                    'airport_name',
                ]
                if self._pdf_output:
                    row.extend([
                        'aan_pdf_path',
                        'aan_pdf_page_number',
                    ])
                if self._text:
                    row.extend([
                        'landing_direction',
                        'locations',
                        'title',
                        'text',
                        'disclaimers',
                        'email',
                        'effective_from',
                        'effective_to',
                    ])
                if self._text_ocr:
                    row.append('used_ocr')
                self._csv.writerow(row)

            # Add 1 to PDF page numbers to transition from 0-based to 1-based
            # page numbering, as is customarily expected by users of PDFs.
            row = [
                aan.cs_pdf_page.parent.display_path, aan.cs_pdf_page.number + 1,
                aan.page_label, aan.faa_id, aan.airport_name
            ]
            if self._pdf_output:
                row.extend([
                    airport_aan_path,
                    airport_aan_page +
                    1 if airport_aan_page is not None else None,
                ])

            if self._text:
                text_tuple = extract_text_from_cs_arrival_alert_notice(
                    aan.cs_pdf_page, allow_ocr=self._text_ocr)
                row.append(text_tuple.landing_direction)
                row.append('|'.join(text_tuple.locations))
                row.append(text_tuple.title)
                row.append(text_tuple.text)
                row.append('|'.join(text_tuple.disclaimers))
                row.append(text_tuple.email)
                row.append(text_tuple.effective_from)
                row.append(text_tuple.effective_to)
            if self._text_ocr:
                row.append(1 if text_tuple.used_ocr else None)

            self._csv.writerow(row)

            return

        if airport_aan_path != self._last_airport_aan_path:
            self._last_airport_aan_path = airport_aan_path
            print(airport_aan_path)


def main(args):
    parser = argparse.ArgumentParser(
        description='Extracts arrival alert notices from chart supplement PDFs')
    parser.add_argument('file',
                        metavar='chart_supplement',
                        nargs='+',
                        help='a chart supplement volume or rear matter PDF, '
                        'or a .zip file or directory containing these PDFs')
    pdf_out_group = parser.add_mutually_exclusive_group()
    pdf_out_group.add_argument(
        '--out-dir',
        metavar='dir',
        help='where to write extracted arrival alert notice PDFs '
        '(default: same directory as chart_supplement)')
    pdf_out_group.add_argument(
        '--no-pdf-output',
        dest='pdf_output',
        action='store_false',
        help='don’t extract output PDFs (requires --csv)')
    parser.add_argument('--no-header-footer',
                        action='store_true',
                        help='remove headers and footers from extracted PDFs')
    metadata_out_group = parser.add_mutually_exclusive_group()
    metadata_out_group.add_argument(
        '-q',
        '--quiet',
        action='store_true',
        help='don’t produce any output on standard output '
        '(default: write output filenames to standard output)')
    metadata_out_group.add_argument(
        '--csv',
        action='store_true',
        help='write arrival alert notice information to standard output in CSV '
        'format')
    parser.add_argument('--text',
                        action='store_true',
                        help='include notice text in CSV (requires --csv)')
    parser.add_argument('--text-ocr',
                        action='store_true',
                        help='use OCR to extract notice text if required '
                        '(requires --text)')
    parsed = parser.parse_args(args)

    if not parsed.pdf_output and not parsed.csv:
        parser.error('--no-pdf-output requires --csv')
    if not parsed.pdf_output and parsed.no_header_footer:
        parser.error('--no-pdf-output and --no-header-footer are incompatible')
    if parsed.text and not parsed.csv:
        parser.error('--csv-text requires --csv')
    if parsed.text_ocr and not parsed.text:
        parser.error('--csv-text-ocr requires --csv-text')

    callback = _Callback(quiet=parsed.quiet,
                         use_csv=parsed.csv,
                         text=parsed.text,
                         text_ocr=parsed.text_ocr,
                         pdf_output=parsed.pdf_output)

    # Match both cover-to-cover volume and DCS “application data” rear matter
    # PDF filename conventions.
    _MEMBER_RE = re.compile(
        r'^((CS_[A-Z]{2,3}_[0-9]{8})|'
        r'([A-Z]{2,3}_rear_[0-9]{2}[A-Z]{3}[0-9]{4}))'
        r'\.pdf$', re.IGNORECASE)

    with requests.Session() as session:
        for cs_path in parsed.file:
            pdf_source = _PdfSource(cs_path, _MEMBER_RE, session)
            for cs_pdf in pdf_source.pdfs():
                extract_cs_arrival_alert_notices(
                    cs_pdf,
                    cs_path,
                    output_pdf=parsed.pdf_output,
                    out_dir=parsed.out_dir,
                    remove_header_footer=parsed.no_header_footer,
                    callback=callback._callback)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
