import time
import re
import pytz
import pandas as pd
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import selenium.common.exceptions as se

AMAZON_BASE = "https://www.amazon.com.tr"
AMAZON_LOGIN_PAGE = (
    "https://www.amazon.com.tr/ap/signin?openid.pape.max_auth_age=0"
    "&openid.return_to=https%3A%2F%2Fwww.amazon.com.tr%2F%3Fref_%3Dnav_ya_signin"
    "&openid.identity=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select"
    "&openid.assoc_handle=trflex&openid.mode=checkid_setup"
    "&openid.claimed_id=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select"
    "&openid.ns=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0"
)

TR_MONTHS = {
    "Ocak": 1, "Şubat": 2, "Mart": 3, "Nisan": 4, "Mayıs": 5, "Haziran": 6,
    "Temmuz": 7, "Ağustos": 8, "Eylül": 9, "Ekim": 10, "Kasım": 11, "Aralık": 12,
}


class AmazonScraper:
    def __init__(self, headless: bool = False):
        self.driver = self._make_driver(headless=headless)
        self.driver.get(AMAZON_LOGIN_PAGE)
        print("Amazon login sayfası açıldı. Giriş yaptıktan sonra scrape_product() çağırabilirsiniz.")

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def _make_driver(self, headless: bool = False) -> webdriver.Chrome:
        opts = webdriver.ChromeOptions()
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
            "Chrome/123.0.0.0 Safari/537.36"
        )
        driver = webdriver.Chrome(options=opts)
        driver.set_page_load_timeout(60)
        return driver

    def quit(self):
        self.driver.quit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrape_product(
        self,
        product_url: str,
        target_n: int = 80,
        max_pages: int = 80,
        wait_timeout: int = 15,
    ) -> pd.DataFrame:
        """Scrape up to target_n most-recent local reviews for a single product URL."""
        asin = self._asin_from_url(product_url)
        if not asin:
            raise ValueError(f"ASIN could not be parsed from URL: {product_url}")

        product_meta = self._get_product_meta(asin, wait_timeout=wait_timeout)
        return self._scrape_reviews(asin, product_meta, target_n, max_pages, wait_timeout)

    # ------------------------------------------------------------------
    # URL helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _asin_from_url(url: str) -> str | None:
        m = re.search(r"/dp/([A-Z0-9]{10})", url)
        return m.group(1) if m else None

    def _build_product_url(self, asin: str) -> str:
        return f"{AMAZON_BASE}/dp/{asin}"

    def _build_review_url(self, asin: str, page: int) -> str:
        ref = "cm_cr_arp_d_viewopt_sr" if page <= 1 else f"cm_cr_arp_d_paging_btm_next_{page}"
        return (
            f"{AMAZON_BASE}/product-reviews/{asin}/ref={ref}"
            f"?ie=UTF8&reviewerType=all_reviews&sortBy=recent&pageNumber={page}"
        )

    # ------------------------------------------------------------------
    # Product metadata
    # ------------------------------------------------------------------

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
        product_description = None
        desc_candidates = [
            "#feature-bullets ul li span.a-list-item",
            "ul.a-unordered-list.a-vertical li span.a-list-item",
            "#productDescription",
        ]
        for sel in desc_candidates:
            try:
                els = self.driver.find_elements(By.CSS_SELECTOR, sel)
                texts = [e.text.strip() for e in els if e.text and e.text.strip()]
                if texts:
                    if sel == "#productDescription" and not bullets:
                        product_description = " ".join(texts).strip()
                        break
                    else:
                        bullets.extend(texts)
            except Exception:
                continue

        if bullets:
            product_description = " | ".join(dict.fromkeys(bullets))

        return {
            "product_title": product_title,
            "product_description": product_description,
            "product_url": self._build_product_url(asin),
        }

    # ------------------------------------------------------------------
    # Review scraping
    # ------------------------------------------------------------------

    def _scrape_reviews(
        self,
        asin: str,
        product_meta: dict,
        target_n: int,
        max_pages: int,
        wait_timeout: int,
    ) -> pd.DataFrame:
        wait = WebDriverWait(self.driver, wait_timeout)
        seen: set = set()
        rows: list = []

        self.driver.get(self._build_review_url(asin, 1))
        last_url = None

        for page in range(1, max_pages + 1):
            try:
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#cm_cr-review_list")))
                wait.until(EC.presence_of_element_located(
                    (By.CSS_SELECTOR, '#cm_cr-review_list [data-hook="review"]')
                ))
                self._wait_reviews_stable(timeout=25)
            except se.TimeoutException:
                break

            if self.driver.current_url == last_url:
                break
            last_url = self.driver.current_url

            local_cards, global_cards = self._find_local_and_global_cards()

            if len(global_cards) > 0 or len(local_cards) == 0:
                break

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
                    cards2 = self.driver.find_elements(
                        By.CSS_SELECTOR,
                        '#cm_cr-review_list li[data-hook="review"], '
                        '#cm_cr-review_list div[data-hook="review"]',
                    )
                    if i >= len(cards2):
                        continue
                    row = self._parse_card(cards2[i], asin, product_meta)
                    rid = row.get("review_id")

                if rid:
                    seen.add(rid)
                rows.append(row)

                if len(seen) >= target_n:
                    return self._to_df(rows, target_n)

            time.sleep(0.6)

            try:
                if self.driver.find_elements(By.CSS_SELECTOR, "li.a-disabled.a-last"):
                    break
                current_first = None
                first_cards = self.driver.find_elements(
                    By.CSS_SELECTOR, '#cm_cr-review_list [data-hook="review"]'
                )
                if first_cards:
                    current_first = first_cards[0]

                next_a = wait.until(EC.element_to_be_clickable((
                    By.CSS_SELECTOR, "li.a-last a, a[data-hook='pagination-next']"
                )))
                self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", next_a)
                try:
                    next_a.click()
                except Exception:
                    self.driver.execute_script("arguments[0].click();", next_a)

                if current_first is not None:
                    wait.until(EC.staleness_of(current_first))
            except se.TimeoutException:
                break

        return self._to_df(rows, target_n)

    # ------------------------------------------------------------------
    # Page helpers
    # ------------------------------------------------------------------

    def _wait_reviews_stable(self, timeout: int = 20, stable_rounds: int = 3, poll: float = 0.25) -> int:
        end = time.time() + timeout
        last = -1
        stable = 0
        while time.time() < end:
            cards = self.driver.find_elements(
                By.CSS_SELECTOR, '#cm_cr-review_list [data-hook="review"]'
            )
            n = len(cards)
            if n == last and n > 0:
                stable += 1
                if stable >= stable_rounds:
                    return n
            else:
                stable = 0
                last = n
            time.sleep(poll)
        return len(self.driver.find_elements(
            By.CSS_SELECTOR, '#cm_cr-review_list [data-hook="review"]'
        ))

    def _find_local_and_global_cards(self):
        container = self.driver.find_element(By.CSS_SELECTOR, "#cm_cr-review_list")
        cards = container.find_elements(
            By.CSS_SELECTOR, 'li[data-hook="review"], div[data-hook="review"]'
        )
        local_headers = container.find_elements(
            By.CSS_SELECTOR, 'h3[data-hook="arp-local-reviews-header"]'
        )
        local_text = local_headers[0].text.strip() if local_headers else ""
        is_local = ("Ülke:" in local_text) or ("Türkiye" in local_text)

        # Some pages do not render the local/global header immediately.
        # In that case, treat visible cards as local to avoid false early stop.
        if not local_headers:
            return cards, []

        return (cards, []) if is_local else ([], cards)

    # ------------------------------------------------------------------
    # Card parsing
    # ------------------------------------------------------------------

    def _parse_card(self, card, asin: str, product_meta: dict) -> dict:
        title = None
        try:
            title = card.find_element(By.CSS_SELECTOR, '[data-hook="review-title"]').text.strip()
        except Exception:
            pass

        rating = self._extract_rating(card)

        body = None
        try:
            body = card.find_element(By.CSS_SELECTOR, '[data-hook="review-body"]').text.strip().replace("\n", " ")
        except Exception:
            pass

        review_date = None
        try:
            date_text = card.find_element(By.CSS_SELECTOR, 'span[data-hook="review-date"]').text.strip()
            review_date = self._parse_date(date_text)
        except Exception:
            pass

        review_id = None
        try:
            review_id = card.get_attribute("id")
        except Exception:
            pass

        collected_at = datetime.now(pytz.timezone("Europe/Istanbul")).strftime("%Y-%m-%d %H:%M")

        return {
            "review_id": review_id,
            "asin": asin,
            "title": title,
            "rating": rating,
            "body": body,
            "review_date": review_date,
            "product_title": (product_meta or {}).get("product_title"),
            "product_description": (product_meta or {}).get("product_description"),
            "product_url": self._build_product_url(asin),
            "source": "amazon_tr",
            "review_summary": None,
            "collected_at": collected_at,
        }

    @staticmethod
    def _extract_rating(card) -> float | None:
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
        if not mon:
            return None
        return datetime(year, mon, day).date().isoformat()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _to_df(rows: list, target_n: int) -> pd.DataFrame:
        df = pd.DataFrame(rows[:target_n])
        if "review_id" in df.columns:
            df = df.drop_duplicates(subset=["review_id"])
        elif not df.empty:
            df = df.drop_duplicates()
        return df
