graph TD
    subgraph Step1_Preprocess [1. 文本预处理]
        A[get_learned_condition] --> B[Tokenize & Parse Attention]
        B --> C[T5 Tokenizer Encode]
        C --> D[生成 t5_tokens, weights, mask]
    end

    subgraph Step2_Loop [2. 分块处理 get_learned_condition_common]
        D --> E[计算 chunk_count]
        E --> F{循环每个 Chunk}
        F --> G[切片 tokens/weights/mask]
        G --> H{use_mask?}
        H -- Yes --> I[创建 t5_attn_mask_chunk]
        H -- No --> J[attention_mask = null]
    end

    subgraph Step3_Compute [3. T5 核心计算]
        I & J --> K[T5Runner.build_graph]
        K --> L[计算 Relative Position Bucket]
        L --> M[T5.forward: Embedding + 24层 Encoder]
        M --> N[final_layer_norm]
    end

    subgraph Step4_PostProcess [4. 后处理与拼接]
        N --> O[应用 chunk_weights 缩放]
        O --> P[Mean-preserving Rescale]
        P --> Q{zero_out_masked?}
        Q -- Yes --> R[mask < 0 处隐层置零]
        Q -- No --> S[保持原样]
        R & S --> T[Append 到 hidden_states_vec]
        T --> U[拼接所有 Chunks]
        U --> V[modify_mask_to_attend_padding]
        V --> W[输出 SDCondition]
    end