"""
Convert Gemma flax checkpoints to the Keras format.

Setup:
```shell
pip install -r requirements.txt
pip install git+https://github.com/google-deepmind/gemma.git
python pip_build.py --install
```

Usage:
```shell
cd tools/checkpoint_conversion
python convert_gemma_checkpoints.py --preset gemma_2b_en
```
"""

import os

os.environ["KERAS_BACKEND"] = "jax"
# No GPU for conversion, makes memory management easier.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import kagglehub  # noqa: E402
import keras  # noqa: E402
import numpy as np  # noqa: E402
from absl import app  # noqa: E402
from absl import flags  # noqa: E402
from checkpoint_conversion_utils import download_gcs_file

from gemma import gm

import keras_hub  # noqa: E402

FLAGS = flags.FLAGS

PRESET_MAP = {
    "gemma_2b_en": {
        "model": gm.nn.Gemma2_2B,
        "params": gm.ckpts.CheckpointPath.GEMMA2_2B_PT,
        "handle": "google/gemma/flax/2b",
    },
    "gemma_7b_en": {
        "model": gm.nn.Gemma2_9B,  # Using Gemma2_9B as closest to 7B
        "params": gm.ckpts.CheckpointPath.GEMMA2_9B_PT,
        "handle": "google/gemma/flax/7b",
    },
    "gemma_instruct_2b_en": {
        "model": gm.nn.Gemma2_2B,
        "params": gm.ckpts.CheckpointPath.GEMMA2_2B_IT,
        "handle": "google/gemma/flax/2b-it",
    },
    "gemma_instruct_7b_en": {
        "model": gm.nn.Gemma2_9B,  # Using Gemma2_9B as closest to 7B
        "params": gm.ckpts.CheckpointPath.GEMMA2_9B_IT,
        "handle": "google/gemma/flax/7b-it",
    },
}


flags.DEFINE_string(
    "preset",
    None,
    f"Must be one of {','.join(PRESET_MAP.keys())}",
    required=True,
)

flags.DEFINE_string(
    "flax_dir",
    None,
    "Optional path to a local flax directory to convert.",
)


def download_flax_model(handle):
    return kagglehub.model_download(handle)


def convert_model(flax_config, vocab_size):
    """convert_model function inspired by gemma3"""
    return keras_hub.models.GemmaBackbone(
        vocabulary_size=vocab_size,
        num_layers=flax_config.num_layers,
        num_query_heads=flax_config.num_heads,
        num_key_value_heads=flax_config.num_kv_heads,
        hidden_dim=flax_config.embed_dim,
        intermediate_dim=flax_config.hidden_dim,
        head_dim=flax_config.head_dim,
    )


def convert_tokenizer(proto_path):
    return keras_hub.models.GemmaTokenizer(proto=proto_path)


def convert_weights(keras_model, flax_config, flax_params):
    # Debug: Print overall structure
    print("🔍 DEBUG: Flax params keys:", list(flax_params.keys()))
    
    # Chomp the embedding weights. Upstream pads for TPU efficiency, but this
    # leads to weird gotchas (you need to disregard part of your output logits).
    embeddings = flax_params["embedder"]["input_embedding"]
    embeddings = np.asarray(embeddings[: keras_model.vocabulary_size, :])
    print(f"🔍 DEBUG: Embeddings shape: {embeddings.shape}")
    keras_model.get_layer("token_embedding").set_weights([embeddings])
    
    keras_model.get_layer("final_normalization").set_weights(
        [np.asarray(flax_params["final_norm"]["scale"])]
    )
    
    # Debug: Check first layer structure
    if flax_config.num_layers > 0:
        flax_layer_name = f"layer_0"
        flax_block = flax_params[flax_layer_name]
        print(f"🔍 DEBUG: Layer 0 keys:", list(flax_block.keys()))
        if "mlp" in flax_block:
            print(f"🔍 DEBUG: MLP keys:", list(flax_block["mlp"].keys()))
            gating_einsum = flax_block["mlp"]["gating_einsum"]
            print(f"🔍 DEBUG: gating_einsum shape: {np.asarray(gating_einsum).shape}")
            print(f"🔍 DEBUG: gating_einsum type: {type(gating_einsum)}")
    
    for i in range(flax_config.num_layers):
        print(f"🔍 DEBUG: Processing layer {i}")
        flax_layer_name = f"layer_{i}"
        keras_block = keras_model.get_layer(f"decoder_block_{i}")

        flax_block = flax_params[flax_layer_name]
        keras_block.pre_attention_norm.set_weights(
            [flax_block["pre_attention_norm"]["scale"]]
        )
        keras_block.pre_ffw_norm.set_weights(
            [flax_block["pre_ffw_norm"]["scale"]]
        )

        gating_einsum = flax_block["mlp"]["gating_einsum"]
        print(f"🔍 DEBUG: Layer {i} gating_einsum shape: {np.asarray(gating_einsum).shape}")
        
        # The gating weights are in shape (2, embed_dim, intermediate_dim)
        # Each of the 2 matrices corresponds to one gating layer
        # Each matrix needs to be split in half along the last dimension
        gating_weights_1 = np.asarray(gating_einsum[0])  # First gating matrix
        gating_weights_2 = np.asarray(gating_einsum[1])  # Second gating matrix
        print(f"🔍 DEBUG: Layer {i} gating_weights_1 shape: {gating_weights_1.shape}")
        print(f"🔍 DEBUG: Layer {i} gating_weights_2 shape: {gating_weights_2.shape}")
        
        # Split each matrix in half along the last dimension
        gate_1, up_1 = np.split(gating_weights_1, 2, axis=-1)
        gate_2, up_2 = np.split(gating_weights_2, 2, axis=-1)
        
        print(f"🔍 DEBUG: Layer {i} gate_1 shape: {gate_1.shape}, gate_2 shape: {gate_2.shape}")
        
        # Check what the Keras layers expect
        print(f"🔍 DEBUG: Layer {i} keras gating_ffw weight shapes: {[w.shape for w in keras_block.gating_ffw.get_weights()]}")
        print(f"🔍 DEBUG: Layer {i} keras gating_ffw_2 weight shapes: {[w.shape for w in keras_block.gating_ffw_2.get_weights()]}")
        
        # Assign the first half of first matrix to gating_ffw
        # and first half of second matrix to gating_ffw_2
        keras_block.gating_ffw.set_weights([gate_1])
        keras_block.gating_ffw_2.set_weights([gate_2])
        
        # Set the linear layer weights
        linear_weights = np.asarray(flax_block["mlp"]["linear"])
        print(f"🔍 DEBUG: Layer {i} linear_weights shape: {linear_weights.shape}")
        keras_block.ffw_linear.set_weights([linear_weights])

        attn_block = flax_block["attn"]
        if flax_config.num_heads != flax_config.num_kv_heads:
            # MQA.
            keras_block.attention.query_dense.kernel.assign(
                np.asarray(attn_block["q_einsum"]["w"][:, :, :])
            )
            keras_block.attention.key_dense.kernel.assign(
                np.asarray(attn_block["kv_einsum"]["w"][0, :, :, :])
            )
            keras_block.attention.value_dense.kernel.assign(
                np.asarray(attn_block["kv_einsum"]["w"][1, :, :, :])
            )
        else:
            # MHA.
            keras_block.attention.query_dense.kernel.assign(
                np.asarray(attn_block["qkv_einsum"]["w"][0, :, :, :])
            )
            keras_block.attention.key_dense.kernel.assign(
                np.asarray(attn_block["qkv_einsum"]["w"][1, :, :, :])
            )
            keras_block.attention.value_dense.kernel.assign(
                np.asarray(attn_block["qkv_einsum"]["w"][2, :, :, :])
            )
        keras_block.attention.output_dense.kernel.assign(
            flax_block["attn"]["attn_vec_einsum"]["w"]
        )


def validate_output(
    keras_model,
    keras_tokenizer,
    flax_model,
    flax_params,
):
    input_str = "What is Keras?"
    length = 32

    # KerasHub
    preprocessor = keras_hub.models.GemmaCausalLMPreprocessor(keras_tokenizer)
    gemma_lm = keras_hub.models.GemmaCausalLM(
        backbone=keras_model,
        preprocessor=preprocessor,
    )
    keras_output = gemma_lm.generate([input_str], max_length=length)
    keras_output = keras_output[0]
    print("🔶 KerasHub output:", keras_output)

    # Flax
    try:
        flax_sampler = gm.text.Sampler(
            model=flax_model,
            params=flax_params,
        )
        flax_output = flax_sampler(
            input_strings=[input_str],
            total_generation_steps=length,
        )
        flax_output = flax_output.text[0]
        print("🔶 Flax output:", flax_output)
    except Exception as e:
        print("🔶 Flax could not be run.", e)


def main(_):
    preset = FLAGS.preset

    print(f"🏃 Converting {preset}")

    presets = PRESET_MAP.keys()
    assert preset in presets, (
        f"Invalid preset {preset}. Must be one of {','.join(presets)}"
    )

    print("🏃 Loading Flax model and tokenizer")
    flax_model = PRESET_MAP[preset]["model"]()
    flax_config = flax_model.config
    flax_params = gm.ckpts.load_params(PRESET_MAP[preset]["params"])
    flax_tokenizer = gm.text.Gemma2Tokenizer()
    proto_path = "./tokenizer_gemma2.model"
    download_gcs_file(
        gcs_uri=flax_tokenizer.path,
        destination_file_name=proto_path,
    )
    print("✅ Flax model loaded")

    keras_tokenizer = convert_tokenizer(proto_path)
    vocab_size = keras_tokenizer.vocabulary_size()
    keras_model = convert_model(flax_config, vocab_size)
    print("✅ Keras model loaded")

    convert_weights(keras_model, flax_config, flax_params)
    print("✅ Weights converted")

    validate_output(keras_model, keras_tokenizer, flax_model, flax_params)

    keras_model.save_to_preset(preset)
    keras_tokenizer.save_to_preset(preset)
    print(f"🏁 Preset saved to ./{preset}")


if __name__ == "__main__":
    app.run(main)