from amazon_scrape import AmazonScraper
from aspect_labeler import AspectLabelingPipeline
from aspect_selector import AspectSentenceSelector
from aspect_summarizer import AspectSummarizer
from evidence_retriever import EvidenceRetriever
from preprocessor import ReviewPreprocessor
from prompt_builder import PromptBuilder


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
    aspect_selector = AspectSentenceSelector(device="cpu", batch_size=32)
    prompt_builder = PromptBuilder()
    summarizer = AspectSummarizer()
    evidence_retriever = EvidenceRetriever(
        sentence_model=aspect_selector.model,
        batch_size=32,
        top_k=5,
        tau=0.45,
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

            df.to_csv("Outputs/output.csv", index=False)
            print(f"Raw reviews saved to output.csv ({len(df)} rows)")

            print("Preprocessing...")
            reviews_df, sentences_df = preprocessor.process(df)

            reviews_df.to_csv("Outputs/reviews.csv", index=False)
            sentences_df.to_csv("Outputs/sentences.csv", index=False)
            print(
                f"Done. reviews.csv={reviews_df.shape}, sentences.csv={sentences_df.shape}"
            )

            print("Labeling aspects...")
            labeled_df = aspect_labeler.predict(
                sentences_df, return_probs=True, return_binary=True
            )
            labeled_df.to_csv("Outputs/labeled_sentences.csv", index=False)
            print(f"labeled_sentences.csv={labeled_df.shape}")

            print("Selecting aspect sentences (relevance + MMR)...")
            mmr_df = aspect_selector.select(labeled_df)
            if mmr_df.empty:
                print("No aspect sentences selected.")
                continue

            mmr_df.to_csv("Outputs/aspect_sentences_mmr.csv", index=False)
            print(f"aspect_sentences_mmr.csv={mmr_df.shape}")

            print("Building prompts...")
            prompt_df = prompt_builder.build(mmr_df)
            print(f"prompt_df={prompt_df.shape}")

            print("Calling teacher + student models...")
            summaries_df = summarizer.summarize_and_save(
                prompt_df, "Outputs/aspect_summaries.csv"
            )

            print("Retrieving evidence for student summaries...")
            evidence_retriever.retrieve_and_save(
                summaries_df=summaries_df,
                mmr_df=mmr_df,
                reviews_df=reviews_df,
                output_path="Outputs/evidence.csv",
            )
    finally:
        scraper.close()


if __name__ == "__main__":
    main()
