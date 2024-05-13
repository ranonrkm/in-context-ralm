def add_retriever_args(parser, retriever_type):
    if retriever_type == "sparse":
        parser.add_argument("--index_name", type=str, default="wikipedia-dpr")
        parser.add_argument("--num_tokens_for_query", type=int, default=32)
        parser.add_argument("--forbidden_titles_path", type=str, default="ralm/retrievers/wikitext103_forbidden_titles.txt")

    elif retriever_type == "dense":
        parser.add_argument("--model_type", type=str, default="bert", choices=["bert", "spider"])
        parser.add_argument("--model_name", type=str, required=True)
        # parser.add_argument("--encoded_files", type=str, required=True)
        # parser.add_argument("--corpus_path", type=str, required=True)
        parser.add_argument("--index_path", type=str, required=True)
        parser.add_argument("--datastore_name", type=str, required=True)

        parser.add_argument("--query_seq_len", type=int, default=128)
        parser.add_argument("--pooling", type=str, default="cls", choices=["cls", "mean"])
        parser.add_argument("--batch_size", type=int, default=2048)
        parser.add_argument("--fp16", action="store_true")
        parser.add_argument("--forbidden_titles_path", type=str, default="ralm/retrievers/wikitext103_forbidden_titles.txt")

    else:
        raise ValueError


def get_retriever(retriever_type, args, tokenizer):
    if retriever_type == "sparse":
        from ralm.retrievers.sparse_retrieval import SparseRetriever
        return SparseRetriever(
            tokenizer=tokenizer,
            index_name=args.index_name,
            forbidden_titles_path=args.forbidden_titles_path,
            num_tokens_for_query=args.num_tokens_for_query,
        )
    elif retriever_type == "dense":
        from ralm.retrievers.dense_retrieval import DenseRetriever
        # return DenseRetriever(
        #     tokenizer=tokenizer,
        #     query_encoder_name=args.query_enc_name, 
        #     query_encoder_class=args.query_enc_cls, 
        #     num_tokens_for_query=args.num_tokens_for_query,
        #     index_name=args.index_name,
        #     forbidden_titles_path=args.forbidden_titles_path,
        # )
        return DenseRetriever(
            query_enc=args.model_name,
            tokenizer=args.tokenizer_name,
            num_tokens_for_query=args.query_seq_len,
            dataset_name=args.datastore_name,
            text_col=args.text_col,
            index_path=args.index_path,
            pooling_strategy=args.pooling,
            gpu_index=True,
        )
    raise ValueError
