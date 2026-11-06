#!/usr/bin/env python3

# Copyright 2026 Mark Mentovai
#
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyright: strict

import argparse
import collections.abc
import contextlib
import csv
import decimal
import os
import re
import typing
import sys
import warnings
import xml.etree.ElementTree


class DataWarning(UserWarning):
    pass


class DataError(ValueError):
    pass


class NavaidKey(typing.NamedTuple):
    # Navaid IDs are reused for different navaid types, so the key used to index
    # by navaids needs to include both the ID and type.

    navaid_id: str
    navaid_type: str


class UnusableComponents(typing.NamedTuple):
    # Defines which components of a navaid are being marked unusable. Some
    # navaids are single-component, but others (such as a VOR-DME) contain
    # multiple components (such as a VOR transmitter and DME transponder), and
    # unusable restrictions may apply to these components individually or
    # grouped together.

    vor: bool = False  # Indicates VOR azimuth service only.
    tacan_azimuth: bool = False
    dme: bool = False
    ndb: bool = False

    @classmethod
    def decode_nasr(
        cls,
        components_s: str,
        navaid_key: NavaidKey,
        remark_s: str,
    ) -> typing.Self | None:
        components_s = re.sub(r'^TACAN(?=$| )', 'TAC', components_s)
        components_s = re.sub(r'(^| )(AZIMUTH)(?=$| )', r'\1AZM', components_s)
        components_s = re.sub(r' AND ', ' & ', components_s)

        vor = False
        tacan_azimuth = False
        dme = False
        ndb = False
        if components_s == 'DME':
            dme = True
        elif components_s == 'NDB':
            ndb = True
        elif components_s == 'NDB/DME':
            dme = True
            ndb = True
        elif components_s == 'TAC':
            tacan_azimuth = True
            dme = True
        elif components_s == 'TAC AZM':
            tacan_azimuth = True
        elif components_s in ('TAC AZM & DME', 'TAC & DME', 'TAC AZM/DME'):
            tacan_azimuth = True
            dme = True
        elif components_s == 'TAC DME':
            dme = True
        elif components_s == 'VOR':
            vor = True
        elif components_s == 'VOR AZM':
            vor = True
        elif components_s in ('VOR AZM & DME', 'VOR & DME', 'VOR/DME'):
            vor = True
            dme = True
        elif components_s == 'TAC DME':
            dme = True
        elif components_s == 'VORTAC':
            vor = True
            tacan_azimuth = True
            dme = True
        elif components_s == 'AZM':
            # In NASR 2611, SHP TACAN has a remark beginning “AZIMUTH UNUSBL”,
            # where it would normaly be “TACAN AZIMUTH UNUSBL”. For navaids that
            # have a single component portion providing azimuth servide, infer
            # the component when only “AZIMUTH” is given.
            if navaid_key.navaid_type in ('VOR', 'VOR/DME'):
                vor = True
            elif navaid_key.navaid_type == 'TACAN':
                tacan_azimuth = True
            else:
                # The navaid type does not provide azimuth service (NDB, DME),
                # or has more than component providing azimuth service (VORTAC).
                warnings.warn(
                    f'cannot infer meaning of {components_s} component of '
                    f'{navaid_key.navaid_type} {navaid_key.navaid_id} in '
                    f'{remark_s!r}',
                    category=DataWarning)
                return None
        else:
            return None

        assert vor or tacan_azimuth or dme or ndb

        # Sanity-check that the components described are actually present for
        # this type of navaid.
        navaid_has_vor = False
        navaid_has_tacan_azimuth = False
        navaid_has_dme = False
        navaid_has_ndb = False
        if navaid_key.navaid_type == 'VOR':
            navaid_has_vor = True
        elif navaid_key.navaid_type == 'VOR/DME':
            navaid_has_vor = True
            navaid_has_dme = True
        elif navaid_key.navaid_type == 'VORTAC':
            navaid_has_vor = True
            navaid_has_tacan_azimuth = True
            navaid_has_dme = True
        elif navaid_key.navaid_type == 'TACAN':
            navaid_has_tacan_azimuth = True
            navaid_has_dme = True
        elif navaid_key.navaid_type == 'DME':
            navaid_has_dme = True
        elif navaid_key.navaid_type == 'NDB':
            navaid_has_ndb = True
        elif navaid_key.navaid_type == 'NDB/DME':
            navaid_has_ndb = True
            navaid_has_dme = True

        # Issue warnings instead of raising exceptions because a small number
        # of violators exist as of NASR 2611.
        if vor and not navaid_has_vor:
            warnings.warn(
                'vor is not a component of '
                f'{navaid_key.navaid_type} {navaid_key.navaid_id} in '
                f'{remark_s!r}',
                category=DataWarning)
            return None
        if tacan_azimuth and not navaid_has_tacan_azimuth:
            warnings.warn(
                'tacan_azimuth is not a component of '
                f'{navaid_key.navaid_type} {navaid_key.navaid_id} in '
                f'{remark_s!r}',
                category=DataWarning)
            return None
        if dme and not navaid_has_dme:
            warnings.warn(
                'dme is not a component of '
                f'{navaid_key.navaid_type} {navaid_key.navaid_id} in '
                f'{remark_s!r}',
                category=DataWarning)
            return None
        if ndb and not navaid_has_ndb:
            warnings.warn(
                'ndb is not a component of '
                f'{navaid_key.navaid_type} {navaid_key.navaid_id} in '
                f'{remark_s!r}',
                category=DataWarning)
            return None

        return cls(vor=vor, tacan_azimuth=tacan_azimuth, dme=dme, ndb=ndb)


class UnusableSectorRadialRange(typing.NamedTuple):
    # Defines a navaid’s unusable sector in terms of radials from the station.
    #
    # This borrows the “radial” terminology used by VOR and TACAN, although it
    # can also be applied to other navaids, including DME and NDB. In these
    # latter cases, “radial” is interpreted as bearings from the station, just
    # as in the VOR and TACAN case. In such cases, the non-VOR/TACAN bearings
    # from the station may be referenced to a colocated VOR or TACAN’s radials,
    # or may be referenced to local magnetic variation, or may be referened to
    # true north.

    # The range of unusuable radials is given by radial_0 and radial_1, which
    # are to be treated inclusively. The units are degrees, in the range 000 to
    # 360 (source data seems to use 000 to 359, but this may not be a
    # guarantee). The unusable sector travels clockwise from radial_0 to
    # radial_1 with respect to the station. A sector composed of single radial
    # marked unusable is expressed with radial_0 = radial_1. A sector that
    # transitions through north (000) will have radial_1 < radial_0.
    radial_0: int
    radial_1: int

    # Whether radial_0 and radial_1 are to be interpreted as magnetic or true
    # bearings from the station. For navaid types that provide radials as a
    # navigation service (VOR and TACAN), the radials should be on the station’s
    # own reference, even if that reference is magnetic and the local magnetic
    # variation does not match the station’s magnetic variation.
    reference: typing.Literal['magnetic'] | typing.Literal['true']


class UnusableSectorDistance(typing.NamedTuple):
    # Defines a navaid’s unusable sector in terms of distance from the station.

    # The unusable sector exists at this minimum distance from the station, and
    # greater, in nautical miles. These are given as “BYD” (beyond) distances in
    # source data.
    min_distance_nm: decimal.Decimal | None

    # The unusable sector exists at this maximum distance from the station, and
    # closer, in nautical miles. These are given as “WI” (within) distances in
    # source data.
    max_distance_nm: decimal.Decimal | None


class UnusableSector(typing.NamedTuple):
    # Defines a navaid’s unusable sector. At least one of the following
    # qualifications must be present. Qualifications that are absent are left at
    # None.

    radial_range: UnusableSectorRadialRange | None
    distance: UnusableSectorDistance | None

    # Defines the unusable sector in terms of the maximum unusable altitude, in
    # feet. These are given as “BLW” (below) altitudes in source data.
    max_altitude_ft: int | None

    @classmethod
    def decode_nasr(
        cls,
        sector_s: str,
        navaid_key: NavaidKey,
        remark_s: str,
    ) -> typing.Self | None:
        radial_range: UnusableSectorRadialRange | None = None
        distance: UnusableSectorDistance | None = None
        max_altitude_ft: int | None = None

        has_distance = False
        has_altitude = False

        while len(sector_s) > 0:
            # Radial specifications are almost always given without a unit, but
            # very infrequently “DEG” or “DEGS” are used.
            #
            # In NASR 2611, BVT VORTAC introduces its radial specification with
            # “BTN” (between), although it is customary to not include any such
            # introduction.
            #
            # In NASR 2611, UTO NDB/DME lists an 045 radial in 2-digit form,
            # “45”, although it is customary to use 3 digits, “045”.
            #
            # In NASR 2611, HCM VORTAC has a sector containing a single radial,
            # 174, given just as “174” and not a range (“174-174”).
            match = re.match(
                r'(?:BTN )?([0-9]{1,3})(T)?'
                r'(?: ?- ?([0-9]{1,3})(T)?)?(?: ?DEGS?)?'
                r'(?: |$)(?P<remain>.*)$', sector_s)
            if match is not None:
                if radial_range is not None:
                    # Multiple radial specifications.
                    return None

                radial_0 = int(match.group(1))
                radial_0_reference = ('true'
                                      if match.group(2) == 'T' else 'magnetic')
                if match.group(3) is not None:
                    radial_1 = int(match.group(3))
                    radial_1_reference = ('true' if match.group(4) == 'T' else
                                          'magnetic')
                else:
                    radial_1 = radial_0
                    radial_1_reference = radial_0_reference

                if radial_0 > 360 or radial_1 > 360:
                    # Invalid radial.
                    return None

                # The form 012T-345T indicates that the datum is true, as
                # opposed to station (or perhaps magnetic). 012-345T is less
                # common, but also indicates true (the T is taken as applying to
                # the radial range, as opposed to individual radials). The form
                # 012T-345 is invalid.
                if radial_0_reference and not radial_1_reference:
                    # Datum disagreement.
                    return None

                radial_range = UnusableSectorRadialRange(
                    radial_0, radial_1, radial_1_reference)

                sector_s = match.group('remain')
                continue

            # Minimum distance specifications are almost always encoded as
            # “BYD”, but very infrequently “BEYOND”. Maximum distance
            # specifications are rare, similarly using “WI” or “WITHIN”. The
            # units are almost always given as “NM” but very infrequently no
            # unit is given.
            match = re.match(
                r'(BYD|BEYOND|WI|WITHIN) ([0-9]+(?:\.[0-9]+)?)(?: ?NM)?'
                r'(?: |$)(?P<remain>.*)$', sector_s)
            if match is not None:
                if has_distance:
                    warnings.warn(
                        'multiple distance specifications at '
                        f'{navaid_key.navaid_type} {navaid_key.navaid_id} in '
                        f'{remark_s!r}',
                        category=DataWarning)
                    return None
                has_distance = True
                min_distance_nm = None
                max_distance_nm = None
                distance_nm = decimal.Decimal(match.group(2))
                if match.group(1) in ('BYD', 'BEYOND'):
                    min_distance_nm = distance_nm
                else:
                    assert match.group(1) in ('WI', 'WITHIN')
                    max_distance_nm = distance_nm
                distance = UnusableSectorDistance(min_distance_nm,
                                                  max_distance_nm)

                sector_s = match.group('remain')
                continue

            # In NASR 2611, BDF TACAN specifies a distance range.
            match = re.match(
                r'BTN ([0-9]+(?:\.[0-9]+)?)(?: ?NM)? AND '
                r'([0-9]+(?:\.[0-9]+)?) ?NM'
                r'(?: |$)(?P<remain>.*)$', sector_s)
            if match is not None:
                if has_distance:
                    warnings.warn(
                        'multiple distance specifications at '
                        f'{navaid_key.navaid_type} {navaid_key.navaid_id} in '
                        f'{remark_s!r}',
                        category=DataWarning)
                    return None
                has_distance = True
                min_distance_nm = decimal.Decimal(match.group(1))
                max_distance_nm = decimal.Decimal(match.group(2))
                if min_distance_nm >= max_distance_nm:
                    raise DataError(
                        'distance disagreement at '
                        f'{navaid_key.navaid_type} {navaid_key.navaid_id} in '
                        f'{remark_s!r}')
                distance = UnusableSectorDistance(min_distance_nm,
                                                  max_distance_nm)

                sector_s = match.group('remain')
                continue

            # Altitude specifications are almost always encoded as “BLW”, but
            # occasionally “BLO”, and very infrequently “BELOW”. The units are
            # almost always given as “FT” but very infrequently “'” (apostrophe)
            # or no unit is given.
            match = re.match(
                r'(?:BLW|BLO|BELOW) ((?:[0-9]+,)?[0-9]+) ?(?:FT|\')?'
                r'(?: |$)(?P<remain>.*)$', sector_s)
            if match is not None:
                if has_altitude:
                    warnings.warn(
                        'multiple altitude specifications at '
                        f'{navaid_key.navaid_type} {navaid_key.navaid_id} in '
                        f'{remark_s!r}',
                        category=DataWarning)
                    return None
                has_altitude = True
                max_altitude_ft = int(match.group(1).replace(',', ''))
                sector_s = match.group('remain')
                continue

            # Very infrequently, “ALL ALTS”, “ALL DSTCS”, or some variation
            # appears, although it does not convey any additional information,
            # as the absence of an altitude or distance specification implies
            # that the restriction is effective at all altitudes or distances.
            # Recognize them and record that an one of these was found (for
            # error detection), but otherwise ignore them.
            match = re.match(
                r'ALL '
                r'(?P<what>ALTS|ALTITUDES|(?:DSTCS|DISTANCES)'
                r'(?: (?:& ?|AND )(?:ALTS|ALTITUDES))?)'
                r'(?: |$)(?P<remain>.*)$', sector_s)
            if match is not None:
                if 'ALT' in match.group('what'):
                    if has_altitude:
                        warnings.warn(
                            'multiple altitude specifications at '
                            f'{navaid_key.navaid_type} {navaid_key.navaid_id} '
                            f'in {remark_s!r}',
                            category=DataWarning)
                        return None
                    has_altitude = True
                if ('DST' in match.group('what') or
                        'DISTANCE' in match.group('what')):
                    if has_distance:
                        warnings.warn(
                            'multiple distance specifications at '
                            f'{navaid_key.navaid_type} {navaid_key.navaid_id} '
                            f'in {remark_s!r}',
                            category=DataWarning)
                        return None
                    has_distance = True

                sector_s = match.group('remain')
                continue

            warnings.warn(
                'unparseable unusable restriction at '
                f'{navaid_key.navaid_type} {navaid_key.navaid_id} in '
                f'{remark_s!r}',
                category=DataWarning)
            return None

        if (radial_range is None and distance is None and
                max_altitude_ft is None):
            # No meaningful restrictions decoded for the sector.
            return None

        return cls(radial_range=radial_range,
                   distance=distance,
                   max_altitude_ft=max_altitude_ft)

    def to_xml(self) -> xml.etree.ElementTree.Element:
        sector_el = xml.etree.ElementTree.Element('sector')

        if self.radial_range is not None:
            xml.etree.ElementTree.SubElement(
                sector_el,
                'radial_range',
                attrib={
                    'radial_0': f'{self.radial_range.radial_0:03d}',
                    'radial_1': f'{self.radial_range.radial_1:03d}',
                    'reference': self.radial_range.reference,
                })

        if self.distance is not None:
            distance_attrs: dict[str, str] = {}
            if self.distance.min_distance_nm is not None:
                distance_attrs['min'] = str(self.distance.min_distance_nm)
            if self.distance.max_distance_nm is not None:
                distance_attrs['max'] = str(self.distance.max_distance_nm)
            assert len(distance_attrs) >= 1
            xml.etree.ElementTree.SubElement(sector_el,
                                             'distance',
                                             attrib=distance_attrs)

        if self.max_altitude_ft is not None:
            xml.etree.ElementTree.SubElement(
                sector_el,
                'altitude',
                attrib={'max': str(self.max_altitude_ft)})

        assert len(sector_el) >= 1

        return sector_el


class UnusableRemark(typing.NamedTuple):
    # Conveys unusable information about a navaid identifying what is unusable
    # (components), where it’s unusable (sectors), and when provided, a reason.

    components: UnusableComponents
    sectors: typing.Sequence[UnusableSector]
    reason: str | None
    is_continuation: bool
    to_be_continued: bool

    @classmethod
    def decode_nasr(
        cls,
        remark_s: str,
        navaid_key: NavaidKey,
    ) -> typing.Self | None:
        orig_remark_s = remark_s

        # Collapse runs of multiple spaces into a single space, and trim leading
        # and trailing spaces.
        remark_s = re.sub(r' +', ' ', remark_s.strip(' '))

        # In NASR 2611, CQX NDB has a structured UNUSBL remark beginning with
        # “UNUSBL” rather than the unusable components. For navaid types with
        # only a single component portion, consider that the implied component
        # if the remark omits it.
        if (remark_s.startswith('UNUSBL ') and
                navaid_key.navaid_type in ('VOR', 'DME', 'NDB')):
            remark_s = f'{navaid_key.navaid_type} {remark_s}'

        # Split the string into the unusable components and unusable sectors.
        # They’re delimited by the string “ UNUSBL ” or, occasionally,
        # “UNUSABLE”. If a remark string doesn’t match, it must not be a
        # structured unusable remark.
        match = re.match(
            r'(?P<components>.+) (UNUSBL|UNUSABLE) ?'
            r'(?P<is_continuation>CONTD:|CONTINUED:)? ?(?P<sectors>.+)$',
            remark_s,
        )
        if match is None:
            return None

        is_continuation = match.group('is_continuation') is not None

        components_s = re.sub(r' PORTION$', '', match.group('components'))
        components = UnusableComponents.decode_nasr(
            components_s,
            navaid_key,
            orig_remark_s,
        )
        if components is None:
            return None

        # Strip the trailing period and any preceding spaces from the sectors
        # specification, so that each sector specification including the final
        # one can be treated identically. In NASR 2611, there are 15 remarks
        # that end in a semicolon instead of a period, so handle those in the
        # same way.
        sectors_s = re.sub(r'[.;,]$', '', match.group('sectors'))

        # In NASR 2611, IL NDB and ZBV VOR/DME have structured unusable remarks
        # that contain reasons for their presence. Extract them to prevent their
        # being mistaken for a part of a sector specification, since that would
        # cause the sector specification to not parse successfully.
        match = re.match(
            r'(?P<sectors>.*)( DUE TO | ?- ?(?=[A-Z]))(?P<reason>.*)$',
            sectors_s)
        if match is not None:
            sectors_s = match.group('sectors')
            reason = match.group('reason')
        else:
            reason = None

        # Sector specifications are normally separated by semicolons, and
        # occasionally by commas. In NASR 2611, ENO VORTAC and HBR VORTAC have
        # at least some sector specifications separated by periods. Canonically,
        # a space will follow the separator character, but there are
        # counter-examples.
        #
        # Although unusual, sector specifications may also contain commas in
        # altitudes (“BLW 5,000 FT”) or periods in distances (“BYD 20.5 NM”). To
        # avoid treating those thousands separators and decimal points as sector
        # specification separators, insist that they be followed by a space when
        # used as a separator. This is not necessary for semicolons used as
        # separators.
        #
        # Very infrequently, “&” or “AND” are used to separate sector
        # specifications, even after a comma. Handle these, but only when
        # following another separator or when followed by a known component of a
        # sector specification, because these strings can also be used within a
        # sector specification (“BTN 25 AND 35 NM”).
        sectors_l = re.split(
            r' ?(?:;|[,.] (?:AND |& ?)?|'
            r'(?:(?: AND | ?& ?)(?=(?:[0-9]+ ?-|BYD|BLO)))) ?',
            sectors_s,
        )

        to_be_continued = (len(sectors_l) > 1 and
                           sectors_l[-1] in ('CONTD', 'CONTINUED'))
        if to_be_continued:
            sectors_l.pop()

        sectors: list[UnusableSector] = []
        for sector_s in sectors_l:
            sector = UnusableSector.decode_nasr(
                sector_s,
                navaid_key,
                orig_remark_s,
            )
            if sector is None:
                return None
            sectors.append(sector)

        if len(sectors) == 0:
            return None
            # No meaningful sectors decoded.

        return cls(
            components=components,
            sectors=tuple(sectors),
            reason=reason,
            is_continuation=is_continuation,
            to_be_continued=to_be_continued,
        )

    def merge(self, other: typing.Self, navaid_key: NavaidKey) -> typing.Self:
        if self.components != other.components:
            raise DataError('components disagree at '
                            f'{navaid_key.navaid_type} {navaid_key.navaid_id}')

        sectors = list(self.sectors)
        sectors.extend(other.sectors)

        # There may be at most one reason, or they must be identical.
        if other.reason is None:
            reason = self.reason
        elif self.reason is not None and self.reason != other.reason:
            raise DataError('reasons disagree at '
                            f'{navaid_key.navaid_type} {navaid_key.navaid_id}')
        else:
            reason = other.reason

        return type(self)(
            components=self.components,
            sectors=tuple(sectors),
            reason=reason,
            is_continuation=self.is_continuation,
            to_be_continued=other.to_be_continued,
        )

    def to_xml(self) -> xml.etree.ElementTree.Element:
        unusable_restriction_el = xml.etree.ElementTree.Element(
            'unusable_restriction')

        components: list[str] = []
        if self.components.vor:
            components.append('vor')
        if self.components.tacan_azimuth:
            components.append('tacan_azimuth')
        if self.components.dme:
            components.append('dme')
        if self.components.ndb:
            components.append('ndb')
        assert len(components) >= 1

        components_el = xml.etree.ElementTree.SubElement(
            unusable_restriction_el, 'components')
        for component in components:
            xml.etree.ElementTree.SubElement(components_el,
                                             'component',
                                             attrib={'type': component})

        sectors_el = xml.etree.ElementTree.SubElement(unusable_restriction_el,
                                                      'sectors')
        for sector in self.sectors:
            sectors_el.append(sector.to_xml())

        assert len(unusable_restriction_el) >= 1

        if self.reason is not None:
            reason_el = xml.etree.ElementTree.SubElement(
                unusable_restriction_el, 'reason')
            reason_el.text = self.reason

        return unusable_restriction_el


class NavaidGeneralRemarks(typing.NamedTuple):
    unusable: dict[UnusableComponents, UnusableRemark]
    unstructured: list[str]

    @classmethod
    def new_empty(cls) -> typing.Self:
        return cls(unusable={}, unstructured=[])


def nasr_nav_rmk_to_general_remarks_dict(
    nasr_nav_rmk_csv_file: typing.TextIO
) -> dict[NavaidKey, NavaidGeneralRemarks]:
    nasr_nav_rmk_csv = csv.DictReader(nasr_nav_rmk_csv_file)

    unprocessed_remarks_d: dict[NavaidKey, list[str]] = {}
    remarks_d: dict[NavaidKey, NavaidGeneralRemarks] = {}
    for row in nasr_nav_rmk_csv:
        if row['REF_COL_NAME'] != 'GENERAL_REMARK':
            continue

        navaid_id = row['NAV_ID']
        navaid_type = row['NAV_TYPE']
        navaid_key = NavaidKey(navaid_id, navaid_type)

        if navaid_key not in unprocessed_remarks_d:
            unprocessed_remarks = []
            unprocessed_remarks_d[navaid_key] = unprocessed_remarks
        else:
            unprocessed_remarks = unprocessed_remarks_d[navaid_key]

        remark_s = row['REMARK']
        unprocessed_remarks.append(remark_s)

        if navaid_key not in remarks_d:
            navaid_general_remarks = NavaidGeneralRemarks.new_empty()
            remarks_d[navaid_key] = navaid_general_remarks
        else:
            navaid_general_remarks = remarks_d[navaid_key]

        unusable_remark = UnusableRemark.decode_nasr(remark_s, navaid_key)
        if unusable_remark is None:
            navaid_general_remarks.unstructured.append(remark_s)
        else:
            components = unusable_remark.components
            navaid_unusable_remarks = navaid_general_remarks.unusable
            if components in navaid_unusable_remarks:
                unusable_remark = (navaid_unusable_remarks[components].merge(
                    unusable_remark, navaid_key))
            navaid_unusable_remarks[components] = unusable_remark

    for navaid_key, navaid_unusable_remarks in remarks_d.items():
        for unusable_remark in navaid_unusable_remarks.unusable.values():
            if (unusable_remark.is_continuation or
                    unusable_remark.to_be_continued):
                # A continuation couldn’t be merged. Rather than presenting some
                # restrictions in structured form and others as unstructured
                # remarks, throw out all of the structured restrictions and go
                # back to just the unstructured ones. Even if they’re less
                # usable, at least all of the information will be in the same
                # place.
                warnings.warn(
                    'unmerged continuation at '
                    f'{navaid_key.navaid_type} {navaid_key.navaid_id}',
                    category=DataWarning)
                navaid_unusable_remarks.unusable.clear()
                navaid_unusable_remarks.unstructured.clear()
                navaid_unusable_remarks.unstructured.extend(
                    unprocessed_remarks_d[navaid_key])
                break

    return remarks_d


def _merge_general_remarks_dict_components(
    remarks_d: dict[NavaidKey, NavaidGeneralRemarks]
) -> dict[NavaidKey, NavaidGeneralRemarks]:
    # To coalesce components having identical unusable restrictions on a
    # per-navaid basis, invert each navaid’s unusable dict so that the
    # restrictions (without components) are keys and the components are values.
    # As the merged inverted dict is built up, coalesce values (components) when
    # identical keys (restrictions) are found. Then, invert the whole thing
    # again to produce a merged unusable dict for the navaid.
    merged_remarks_d: dict[NavaidKey, NavaidGeneralRemarks] = {}
    for navaid_key, navaid_remarks in remarks_d.items():
        # Build an inverted dict. Don’t use UnusableRemark as the key, because
        # it includes UnusableComponents.
        inverted_unusable_remarks: dict[tuple[typing.Sequence[UnusableSector],
                                              str | None, bool, bool],
                                        UnusableComponents] = {}
        for unusable_remark in navaid_remarks.unusable.values():
            components = unusable_remark.components
            inverted_unusable_remark = (unusable_remark.sectors,
                                        unusable_remark.reason,
                                        unusable_remark.is_continuation,
                                        unusable_remark.to_be_continued)
            if inverted_unusable_remark in inverted_unusable_remarks:
                # The restriction is identical to one that was already recorded
                # for different UnusableComponents. Merge the two
                # UnusableComponents into one.
                existing_components = inverted_unusable_remarks[
                    inverted_unusable_remark]
                components = UnusableComponents(
                    vor=components.vor or existing_components.vor,
                    tacan_azimuth=(components.tacan_azimuth or
                                   existing_components.tacan_azimuth),
                    dme=components.dme or existing_components.dme,
                    ndb=components.ndb or existing_components.ndb,
                )
            inverted_unusable_remarks[inverted_unusable_remark] = components

        # Invert the dict again, reintroducing UnusableRemark values that
        # contain UnusableComponents.
        rebuilt_unusable_remarks: dict[UnusableComponents, UnusableRemark] = {}
        for ((sectors, reason, is_continuation, to_be_continued),
             components) in inverted_unusable_remarks.items():
            unusable_remark = UnusableRemark(components=components,
                                             sectors=sectors,
                                             reason=reason,
                                             is_continuation=is_continuation,
                                             to_be_continued=to_be_continued)
            rebuilt_unusable_remarks[components] = unusable_remark

        merged_remarks_d[navaid_key] = NavaidGeneralRemarks(
            unusable=rebuilt_unusable_remarks,
            unstructured=navaid_remarks.unstructured)

    return merged_remarks_d


def general_remarks_dict_to_xml(
    remarks_d: dict[NavaidKey, NavaidGeneralRemarks]
) -> xml.etree.ElementTree.Element:
    navaid_remarks_el = xml.etree.ElementTree.Element('navaid_remarks')
    for (navaid_id, navaid_type), navaid_remarks in remarks_d.items():
        navaid_el = xml.etree.ElementTree.SubElement(navaid_remarks_el,
                                                     'navaid',
                                                     attrib={
                                                         'id': navaid_id,
                                                         'type': navaid_type
                                                     })

        if len(navaid_remarks.unstructured):
            general_remarks_el = xml.etree.ElementTree.SubElement(
                navaid_el, 'general_remarks')
            for general_remark_s in navaid_remarks.unstructured:
                general_remark_el = xml.etree.ElementTree.SubElement(
                    general_remarks_el, 'general_remark')
                general_remark_el.text = general_remark_s

        if len(navaid_remarks.unusable):
            unusable_restrictions_el = xml.etree.ElementTree.SubElement(
                navaid_el, 'unusable_restrictions')
            for unusable_remark in navaid_remarks.unusable.values():
                unusable_restrictions_el.append(unusable_remark.to_xml())

    return navaid_remarks_el


@contextlib.contextmanager
def _open_or_use(
        path: os.PathLike[str] | str | None,
        use: typing.TextIO,
        mode: typing.Literal['r', 'w'],
        newline: str | None = None) -> collections.abc.Generator[typing.TextIO]:
    if path is None or path == '-':
        yield use
    else:
        with open(path, mode, newline=newline) as file:
            yield file


def main(args: typing.Sequence[str]) -> int | None:
    parser = argparse.ArgumentParser(
        description='Extract NASR navaid unusable remarks in a structured form.'
    )
    parser.add_argument(
        '--merge',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='merge identical restrictions for multiple components')
    parser.add_argument('--format',
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help='indent XML output')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--quiet', action='store_true', help='suppress warnings')
    group.add_argument('--strict',
                       action='store_true',
                       help='treat warnings as errors')
    parser.add_argument('nav_rmk_csv_path',
                        help='path to NASR NAV_RMK.csv input, - for stdin,')
    parser.add_argument('xml_out_path',
                        nargs='?',
                        help='path to XML output, - for stdout')
    parsed = parser.parse_args(args)

    with warnings.catch_warnings():
        if parsed.quiet:
            warnings.simplefilter('ignore')
        elif parsed.strict:
            warnings.simplefilter('error')

        with _open_or_use(
                parsed.nav_rmk_csv_path,
                sys.stdin,
                'r',
        ) as nav_rmk_csv_file:
            remarks_d = nasr_nav_rmk_to_general_remarks_dict(nav_rmk_csv_file)

        if parsed.merge:
            remarks_d = _merge_general_remarks_dict_components(remarks_d)

        xml_root = general_remarks_dict_to_xml(remarks_d)
        xml_tree = xml.etree.ElementTree.ElementTree(xml_root)

        if parsed.format:
            xml.etree.ElementTree.indent(xml_tree)

        with _open_or_use(
                parsed.xml_out_path,
                sys.stdout,
                'w',
                newline='\n',
        ) as xml_out_file:
            xml_tree.write(xml_out_file,
                           encoding='unicode',
                           xml_declaration=True)

            # xml.etree.ElementTree.ElementTree.write doesn’t conclude its
            # output with a trailing newline.
            xml_out_file.write('\n')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
