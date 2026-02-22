from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from maternal_mortality_dashboard.config import Settings
from maternal_mortality_dashboard.exceptions import DataIngestionError

logger = logging.getLogger(__name__)


class WorldBankClient:
    """HTTP client for World Bank API extraction with retry + pagination support."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.session = requests.Session()
        retry_strategy = Retry(
            total=int(settings.request_retries),
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _request_page(self, endpoint: str, params: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        url = urljoin(str(self.settings.world_bank_api_base).rstrip("/") + "/", endpoint.lstrip("/"))
        try:
            response = self.session.get(
                url=url,
                params=params,
                timeout=int(self.settings.request_timeout_seconds),
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise DataIngestionError(f"World Bank request failed for endpoint {endpoint}") from exc
        except ValueError as exc:
            raise DataIngestionError(f"Invalid JSON payload returned by endpoint {endpoint}") from exc

        if not isinstance(payload, list) or len(payload) != 2:
            raise DataIngestionError(f"Unexpected World Bank response shape for endpoint {endpoint}")

        metadata = payload[0] or {}
        rows = payload[1] or []
        return metadata, rows

    def fetch_indicator(self, indicator_id: str, start_year: int, end_year: int) -> pd.DataFrame:
        endpoint = f"/country/all/indicator/{indicator_id}"
        page = 1
        pages = 1
        all_rows: list[dict[str, Any]] = []

        while page <= pages:
            metadata, rows = self._request_page(
                endpoint=endpoint,
                params={
                    "format": "json",
                    "per_page": int(self.settings.world_bank_per_page),
                    "date": f"{start_year}:{end_year}",
                    "page": page,
                },
            )
            pages = int(metadata.get("pages") or 1)
            all_rows.extend(rows)
            page += 1

        if not all_rows:
            raise DataIngestionError(f"No rows returned for indicator {indicator_id}")

        records = []
        for row in all_rows:
            country = row.get("country") or {}
            indicator = row.get("indicator") or {}
            value = row.get("value")
            year = row.get("date")
            iso3 = row.get("countryiso3code")

            if not iso3 or iso3 == "":
                continue

            records.append(
                {
                    "country_iso3": iso3,
                    "country_name": country.get("value"),
                    "indicator_id": indicator.get("id"),
                    "indicator_name": indicator.get("value"),
                    "year": int(year) if year is not None else None,
                    "value": float(value) if value is not None else None,
                }
            )

        frame = pd.DataFrame.from_records(records)
        if frame.empty:
            raise DataIngestionError(f"Indicator {indicator_id} returned no country-level records")

        return frame

    def fetch_country_metadata(self) -> pd.DataFrame:
        endpoint = "/country"
        page = 1
        pages = 1
        all_rows: list[dict[str, Any]] = []

        while page <= pages:
            metadata, rows = self._request_page(
                endpoint=endpoint,
                params={
                    "format": "json",
                    "per_page": int(self.settings.world_bank_per_page),
                    "page": page,
                },
            )
            pages = int(metadata.get("pages") or 1)
            all_rows.extend(rows)
            page += 1

        if not all_rows:
            raise DataIngestionError("No country metadata returned from World Bank")

        records = []
        for row in all_rows:
            records.append(
                {
                    "country_iso3": row.get("id"),
                    "region": (row.get("region") or {}).get("value"),
                    "income_level": (row.get("incomeLevel") or {}).get("value"),
                    "lending_type": (row.get("lendingType") or {}).get("value"),
                }
            )

        frame = pd.DataFrame.from_records(records).dropna(subset=["country_iso3"])
        if frame.empty:
            raise DataIngestionError("Country metadata frame was empty after normalization")

        logger.info("Fetched country metadata for %s entities", len(frame))
        return frame
