import time
import re
import pytz
import pandas as pd
from datetime import datetime
from pathlib import Path
import threading

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import selenium.common.exceptions as se

AMAZON_BASE = "https://www.amazon.com.tr"
AMAZON_LOGIN_URL = (
    "https://www.amazon.com.tr/"
)
DATA_DIR = Path(__file__).resolve().parent / "data"

TR_MONTHS = {
    "Ocak": 1, "Şubat": 2, "Mart": 3, "Nisan": 4, "Mayıs": 5, "Haziran": 6,
    "Temmuz": 7, "Ağustos": 8, "Eylül": 9, "Ekim": 10, "Kasım": 11, "Aralık": 12,
}


class AmazonScraper:
    def __init__(self, headless: bool = False, detach: bool = True):
        self.driver = self._make_driver(headless=headless, detach=detach)
        self.driver.get(AMAZON_LOGIN_URL)
        print("Amazon login page opened. Please log in, then call scrape(asin_or_url).")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrape(self, asin_or_url: str, target_n: int = 80, max_pages: int = 80, wait_timeout: int = 15) -> pd.DataFrame:
        """Scrape reviews for a single product. Pass an ASIN or full Amazon URL."""
        asin = self._extract_asin(asin_or_url)
        if not asin:
            raise ValueError(f"Could not extract a valid ASIN from: {asin_or_url!r}")

        print(f"Scraping ASIN: {asin}")
        df = self._scrape_product(asin, target_n=target_n, max_pages=max_pages, wait_timeout=wait_timeout)
        output_path = self._save_scraped_data(asin, df)
        print(f"  -> {len(df)} reviews collected")
        print(f"  -> scraped data saved to {output_path}")
        return df

    def close(self):
        try:
            self.driver.quit()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_driver(headless: bool = False, detach: bool = True) -> webdriver.Chrome:
        opts = webdriver.ChromeOptions()
        if detach:
            opts.add_experimental_option("detach", True)
        if headless:
            opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--window-size=1400,900")
        opts.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        )
        driver = webdriver.Chrome(options=opts)
        driver.set_page_load_timeout(60)
        return driver

    @staticmethod
    def _extract_asin(value: str) -> str | None:
        value = value.strip()
        # Pure ASIN (10 alphanumeric chars)
        if re.fullmatch(r"[A-Z0-9]{10}", value, re.IGNORECASE):
            return value.upper()
        # From URL: /dp/ASIN or /product-reviews/ASIN
        m = re.search(r"/(?:dp|product-reviews|gp/product)/([A-Z0-9]{10})", value, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        return None

    @staticmethod
    def _save_scraped_data(asin: str, df: pd.DataFrame) -> Path:
        DATA_DIR.mkdir(exist_ok=True)
        output_path = DATA_DIR / f"{asin}_data.csv"
        df.to_csv(output_path, index=False)
        return output_path

    def _build_review_url(self, asin: str, page: int) -> str:
        ref = "cm_cr_arp_d_viewopt_sr" if page <= 1 else f"cm_cr_arp_d_paging_btm_next_{page}"
        return (
            f"{AMAZON_BASE}/product-reviews/{asin}/ref={ref}"
            f"?ie=UTF8&reviewerType=all_reviews&sortBy=recent&pageNumber={page}"
        )

    def _build_product_url(self, asin: str) -> str:
        return f"{AMAZON_BASE}/dp/{asin}"

    # ------------------------------------------------------------------
    # Page interactions
    # ------------------------------------------------------------------

    def _wait_reviews_stable(self, timeout: int = 20, stable_rounds: int = 3, poll: float = 0.25) -> int:
        end = time.time() + timeout
        last, stable = -1, 0
        while time.time() < end:
            cards = self.driver.find_elements(By.CSS_SELECTOR, '#cm_cr-review_list [data-hook="review"]')
            n = len(cards)
            if n == last and n > 0:
                stable += 1
                if stable >= stable_rounds:
                    return n
            else:
                stable, last = 0, n
            time.sleep(poll)
        return len(self.driver.find_elements(By.CSS_SELECTOR, '#cm_cr-review_list [data-hook="review"]'))

    def _get_product_meta(self, asin: str, wait_timeout: int = 15) -> dict:
        wait = WebDriverWait(self.driver, wait_timeout)
        self.driver.get(self._build_product_url(asin))

        product_title = None
        try:
            el = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#productTitle")))
            product_title = el.text.strip()
        except Exception:
            pass

        bullets = []
        for sel in [
            "#feature-bullets ul li span.a-list-item",
            "ul.a-unordered-list.a-vertical li span.a-list-item",
            "#productDescription",
        ]:
            try:
                els = self.driver.find_elements(By.CSS_SELECTOR, sel)
                texts = [e.text.strip() for e in els if e.text and e.text.strip()]
                if texts:
                    if sel == "#productDescription" and not bullets:
                        bullets = texts
                    else:
                        bullets.extend(texts)
                    if sel != "#productDescription":
                        break
            except Exception:
                continue

        product_description = " | ".join(dict.fromkeys(bullets)) if bullets else None

        return {
            "product_title": product_title,
            "product_description": product_description,
            "product_url": self._build_product_url(asin),
        }

    @staticmethod
    def _is_foreign_card(card) -> bool:
        # Primary marker: foreign reviews wrap content in <div id="customer_review_foreign-...">
        if card.find_elements(By.CSS_SELECTOR, '[id^="customer_review_foreign-"]'):
            return True
        # Backup markers: translated-content span or "Translate to Turkish" link
        if card.find_elements(By.CSS_SELECTOR, 'span.cr-original-review-content'):
            return True
        if card.find_elements(By.CSS_SELECTOR, '[data-hook="cr-translate-this-review-link"]'):
            return True
        return False

    def _find_local_and_global_cards(self):
        container = self.driver.find_element(By.CSS_SELECTOR, "#cm_cr-review_list")
        cards = container.find_elements(By.CSS_SELECTOR, 'li[data-hook="review"], div[data-hook="review"]')

        local_cards, global_cards = [], []
        for card in cards:
            if self._is_foreign_card(card):
                global_cards.append(card)
            else:
                local_cards.append(card)
        return local_cards, global_cards

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_text(el) -> str:
        try:
            return el.text.strip()
        except Exception:
            return ""

    @staticmethod
    def _parse_rating(card) -> float | None:
        try:
            txt = card.get_attribute("innerText") or card.get_attribute("textContent") or ""
            txt = txt.replace("\u00a0", " ").strip()
        except Exception:
            txt = (card.text or "").strip()

        if not txt:
            return None

        for pattern in [
            r"(\d+(?:[.,]\d+)?)\s*yıldız\s*üzerinden\s*(\d+(?:[.,]\d+)?)",
            r"(\d+(?:[.,]\d+)?)\s*üzerinden\s*(\d+(?:[.,]\d+)?)",
        ]:
            m = re.search(pattern, txt, flags=re.IGNORECASE)
            if m:
                return float(m.group(2).replace(",", "."))

        m = re.search(r"(\d+(?:[.,]\d+)?)\s*yıldız", txt, flags=re.IGNORECASE)
        if m:
            return float(m.group(1).replace(",", "."))

        return None

    @staticmethod
    def _parse_date(date_text: str) -> str | None:
        if not date_text:
            return None
        m = re.search(r"(\d{1,2})\s+([A-Za-zÇĞİÖŞÜçğıöşü]+)\s+(\d{4})", date_text)
        if not m:
            return None
        day, mon_name, year = int(m.group(1)), m.group(2).strip(), int(m.group(3))
        mon = TR_MONTHS.get(mon_name)
        return datetime(year, mon, day).date().isoformat() if mon else None

    def _parse_card(self, card, asin: str, product_meta: dict) -> dict:
        title = None
        try:
            title = self._safe_text(card.find_element(By.CSS_SELECTOR, '[data-hook="review-title"]'))
        except Exception:
            pass

        body = None
        try:
            body = self._safe_text(card.find_element(By.CSS_SELECTOR, '[data-hook="review-body"]')).replace("\n", " ")
        except Exception:
            pass

        review_date = None
        try:
            review_date = self._parse_date(
                self._safe_text(card.find_element(By.CSS_SELECTOR, 'span[data-hook="review-date"]'))
            )
        except Exception:
            pass

        review_id = None
        try:
            review_id = card.get_attribute("id")
        except Exception:
            pass

        tz = pytz.timezone("Europe/Istanbul")
        return {
            "review_id": review_id,
            "asin": asin,
            "title": title,
            "rating": self._parse_rating(card),
            "body": body,
            "review_date": review_date,
            "product_title": (product_meta or {}).get("product_title"),
            "product_description": (product_meta or {}).get("product_description"),
            "product_url": self._build_product_url(asin),
            "source": "amazon_tr",
            "review_summary": None,
            "collected_at": datetime.now(tz).strftime("%Y-%m-%d %H:%M"),
        }

    # ------------------------------------------------------------------
    # Core scraping logic
    # ------------------------------------------------------------------

    def _scrape_product(self, asin: str, target_n: int, max_pages: int, wait_timeout: int) -> pd.DataFrame:
        wait = WebDriverWait(self.driver, wait_timeout)
        seen, rows = set(), []

        try:
            product_meta = self._get_product_meta(asin, wait_timeout=wait_timeout)
        except Exception:
            product_meta = {"product_title": None, "product_description": None, "product_url": self._build_product_url(asin)}

        self.driver.get(self._build_review_url(asin, 1))
        stagnant_rounds = 0

        for page in range(1, max_pages + 1):
            try:
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#cm_cr-review_list")))
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '#cm_cr-review_list [data-hook="review"]')))
                self._wait_reviews_stable(timeout=25, stable_rounds=4)
            except se.TimeoutException:
                break

            local_cards, _ = self._find_local_and_global_cards()

            if len(local_cards) == 0:
                break

            page_had_new = False

            for i, card in enumerate(local_cards):
                try:
                    rid = card.get_attribute("id")
                except Exception:
                    rid = None

                if rid and rid in seen:
                    continue

                try:
                    row = self._parse_card(card, asin, product_meta)
                except se.StaleElementReferenceException:
                    fresh = self.driver.find_elements(
                        By.CSS_SELECTOR,
                        '#cm_cr-review_list li[data-hook="review"], #cm_cr-review_list div[data-hook="review"]'
                    )
                    if i >= len(fresh):
                        continue
                    row = self._parse_card(fresh[i], asin, product_meta)
                    rid = row.get("review_id")

                if rid:
                    seen.add(rid)
                rows.append(row)
                page_had_new = True

                if len(seen) >= target_n:
                    df = pd.DataFrame(rows)
                    return df.drop_duplicates(subset=["review_id"]).head(target_n)

            if page_had_new:
                stagnant_rounds = 0
            else:
                stagnant_rounds += 1
                if stagnant_rounds >= 2:
                    print(f"  -> no new reviews on page {page}, stopping")
                    break

            time.sleep(0.6)

            if not self._navigate_to_next_page():
                break

        df = pd.DataFrame(rows[:target_n])
        if "review_id" in df.columns:
            df = df.drop_duplicates(subset=["review_id"])
        else:
            df = df.drop_duplicates()
        return df

    def _navigate_to_next_page(self) -> bool:
        url_before = self.driver.current_url
        count_before = len(
            self.driver.find_elements(By.CSS_SELECTOR, '#cm_cr-review_list [data-hook="review"]')
        )
        wait = WebDriverWait(self.driver, 15)

        # Standard next-page button
        try:
            if not self.driver.find_elements(By.CSS_SELECTOR, "li.a-disabled.a-last"):
                candidates = self.driver.find_elements(By.CSS_SELECTOR, "li.a-last a")
                if candidates:
                    a = candidates[0]
                    self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", a)
                    try:
                        a.click()
                    except Exception:
                        self.driver.execute_script("arguments[0].click();", a)

                    try:
                        wait.until(lambda d: d.current_url != url_before)
                        return True
                    except se.TimeoutException:
                        return False
        except Exception:
            pass

        # "Show more" button (appends reviews without changing URL)
        try:
            show_more = [
                el for el in self.driver.find_elements(By.CSS_SELECTOR, 'a[data-hook="show-more-button"]')
                if el.is_displayed()
            ]
            if show_more:
                a = show_more[0]
                self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", a)
                try:
                    a.click()
                except Exception:
                    self.driver.execute_script("arguments[0].click();", a)

                try:
                    wait.until(
                        lambda d: len(d.find_elements(By.CSS_SELECTOR, '#cm_cr-review_list [data-hook="review"]')) > count_before
                    )
                    return True
                except se.TimeoutException:
                    current = len(self.driver.find_elements(By.CSS_SELECTOR, '#cm_cr-review_list [data-hook="review"]'))
                    return current > count_before
        except Exception:
            pass

        return False
