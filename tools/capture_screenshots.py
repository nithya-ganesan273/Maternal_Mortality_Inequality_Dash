"""
Capture dashboard screenshots for the README.

Every image under ``assets/screenshots/`` is produced by this script, so the
documentation can be refreshed after a UI change instead of drifting out of date.

Usage::

    # terminal 1
    python scripts/run_dashboard.py
    # terminal 2
    python tools/capture_screenshots.py

Requires Playwright (``pip install playwright``). It drives the Chrome already
installed on the machine rather than downloading a bundled browser; pass
``--chromium`` to use Playwright's own build instead.

Dash renders its figures client-side over a callback round trip, so each shot
waits for an actual rendered Plotly canvas rather than the load event.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCREENSHOTS = ROOT / "assets" / "screenshots"

WIDTH = 1600
SCALE = 2

# name -> (tab value, selector to wait for, viewport height, scroll px, drag slider?)
#
# The scenario shot drags a slider before capturing: on first paint the sliders
# sit at the selected country-year's observed values, so baseline and adjusted
# are identical and the card reads "+0.0%" - which makes the feature look inert
# in a screenshot even though it is working.
SHOTS = {
    "overview": ("dashboard-tab", "#map-graph .js-plotly-plot", 1150, 0, False),
    "scenario": ("dashboard-tab", "#scenario-model-note", 900, 380, True),
    "inequality": ("dashboard-tab", "#inequality-graph .js-plotly-plot", 1100, 1150, False),
    "drivers": ("dashboard-tab", "#scatter-graph .js-plotly-plot", 1100, 2250, False),
    "methods": ("methods-tab", "#content-tabs", 1250, 0, False),
}

# Fraction along the slider rail to click (skilled birth attendance -> ~90%).
SLIDER_TARGET = 0.9


def capture(base_url: str, use_chromium: bool) -> int:
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeout
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright is not installed. Run: pip install playwright", file=sys.stderr)
        return 1

    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    written = 0

    with sync_playwright() as playwright:
        launch = {"headless": True}
        if not use_chromium:
            launch["channel"] = "chrome"
        browser = playwright.chromium.launch(**launch)

        for name, (tab, selector, height, scroll, drag) in SHOTS.items():
            print(f"  {name:11s} tab={tab:14s} ({WIDTH}x{height})")
            page = browser.new_page(
                viewport={"width": WIDTH, "height": height}, device_scale_factor=SCALE
            )
            page.goto(base_url, wait_until="networkidle", timeout=60_000)

            # Tabs are client-side state, not routes, so the tab must be clicked.
            if tab != "dashboard-tab":
                page.click(f'div.tab[data-value="{tab}"], #content-tabs div:has-text("Methods")')
                page.wait_for_timeout(1_500)

            try:
                page.wait_for_selector(selector, timeout=40_000)
            except PlaywrightTimeout:
                print(f"    warning: {selector!r} never appeared", file=sys.stderr)
            # Plotly animates on first paint; give the callbacks time to settle.
            page.wait_for_timeout(3_500)

            if drag:
                rail = page.locator("#scenario-skilled-birth-slider .rc-slider-rail").first
                box = rail.bounding_box()
                if box:
                    page.mouse.click(
                        box["x"] + box["width"] * SLIDER_TARGET,
                        box["y"] + box["height"] / 2,
                    )
                    page.wait_for_timeout(2_500)  # wait for the callback round trip
                else:
                    print("    warning: slider rail not found", file=sys.stderr)

            if scroll:
                page.mouse.wheel(0, scroll)
                page.wait_for_timeout(1_500)

            page.screenshot(path=str(SCREENSHOTS / f"{name}.png"))
            page.close()
            written += 1

        browser.close()

    print(f"\nWrote {written} screenshots to {SCREENSHOTS}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8050", help="dashboard base URL")
    parser.add_argument(
        "--chromium",
        action="store_true",
        help="use Playwright's bundled Chromium instead of installed Chrome",
    )
    arguments = parser.parse_args()
    return capture(arguments.url.rstrip("/"), arguments.chromium)


if __name__ == "__main__":
    sys.exit(main())
