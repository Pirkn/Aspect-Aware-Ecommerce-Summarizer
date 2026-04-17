from amazon_scrape import AmazonScraper
from aspect_labeler import AspectLabelingPipeline
from preprocessor import ReviewPreprocessor


def main():
    print("=" * 60)
    print("Loading models...")
    print("=" * 60)
    preprocessor = ReviewPreprocessor()
    aspect_labeler = AspectLabelingPipeline(
        model_dir="aspect-labeler",
        text_col="sentence_clean",
        batch_size=32,
        device="cpu",
    )
    print("=" * 60)
    print("All models loaded. Starting scraper...")
    print("=" * 60)

    scraper = AmazonScraper()

    try:
        while True:
            user_input = input("\nEnter the ASIN or URL (or 'q' to quit): ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"q", "quit", "exit"}:
                break

            try:
                df = scraper.scrape(user_input)
            except Exception as e:
                print(f"Scrape failed: {e}")
                continue

            if df is None or df.empty:
                print("No reviews collected; skipping.")
                continue

            df.to_csv("output.csv", index=False)
            print(f"Raw reviews saved to output.csv ({len(df)} rows)")

            print("Preprocessing...")
            reviews_df, sentences_df = preprocessor.process(df)

            reviews_df.to_csv("reviews.csv", index=False)
            sentences_df.to_csv("sentences.csv", index=False)
            print(
                f"Done. reviews.csv={reviews_df.shape}, sentences.csv={sentences_df.shape}"
            )

            print("Labeling aspects...")
            labeled_df = aspect_labeler.predict(
                sentences_df, return_probs=True, return_binary=True
            )
            labeled_df.to_csv("labeled_sentences.csv", index=False)
            print(f"labeled_sentences.csv={labeled_df.shape}")
    finally:
        scraper.close()


if __name__ == "__main__":
    main()
