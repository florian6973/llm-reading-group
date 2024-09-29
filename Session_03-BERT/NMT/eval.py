import torch

from model import subsequent_mask, Batch, make_model
from data import create_dataloaders, load_all_vocab

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


def transition_fn(model, memory, src, src_mask, ys, beam_width):

    # print(state)

    out = model.decode(
        memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
    )
    probs = model.generator(out[:, -1])

    probabilities, idx = probs.topk(
        k = beam_width, 
        axis = -1
    )

    # print(probabilities.shape)
    # print(idx.shape, idx, idx[:, 0]
    # exit()
    
    for i in range(beam_width):
        yield probabilities[:, i].unsqueeze(-1),  torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(idx[:, i].data[0])], dim=1
        )
#torch.cat((ys, idx[:, i].unsqueeze(-1)), dim=1) 

def score_fn(action):
    return action


# log likelihood  avoid underflow when computing probabilities of long sequences.
# https://hussainwali.medium.com/simple-implementation-of-beam-search-in-python-64b2d3e2fd7e
def beam_search_abs(model, src, src_mask, max_len, start_symbol, transition_fn, score_fn, beam_width):
    # `start`: the initial state
    # `transition_fn`: a function that takes a state and returns a list of (action, next_state) pairs
    # `score_fn`: a function that takes an action and returns a score
    # `beam_width`: the number of candidates to keep at each step
    # `max_len`: the maximum length of the output sequence
    
    # Initialize the beam with the start state
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    memory = model.encode(src, src_mask)
    beam = [(ys, [], 0)]
    
    # Iterate until we reach the maximum length or run out of candidates
    for i in range(max_len):
        candidates = []
        
        # Generate new candidates by expanding each current candidate
        for state, seq, score in beam:
            for action, next_state in transition_fn(model, memory, src, src_mask, state, beam_width):
                new_seq = seq + [action]
                new_score = score + score_fn(action)
                candidates.append((next_state, new_seq, new_score))
        # print(candidates)
                
        # Select the top `beam_width` candidates based on their scores
        beam = sorted(candidates, key=lambda x: x[2], reverse=True)[:beam_width]
        
    # Return the sequence with the highest score
    return max(beam, key=lambda x: x[2])[0][0]

def check_outputs(
    valid_dataloader,
    model,
    vocab_src,
    vocab_tgt,
    n_examples=15,
    pad_idx=2,
    eos_string="</s>",
):
    results = [()] * n_examples
    for idx in range(n_examples):
        print("\nExample %d ========\n" % idx)
        b = next(iter(valid_dataloader))
        rb = Batch(b[0], b[1], pad_idx)
        # greedy_decode(model, rb.src, rb.src_mask, 64, 0)[0]

        src_tokens = [
            vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx
        ]
        tgt_tokens = [
            vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx
        ]

        print(
            "Source Text (Input)        : "
            + " ".join(src_tokens).replace("\n", "")
        )
        print(
            "Target Text (Ground Truth) : "
            + " ".join(tgt_tokens).replace("\n", "")
        )
        # model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]
        model_out = beam_search_abs(model, rb.src, rb.src_mask,
                                    72, 0, transition_fn, score_fn, 5)
        # print(model_out)
        model_txt = (
            " ".join(
                [vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]
            ).split(eos_string, 1)[0]
            + eos_string
        )
        print("Model Output               : " + model_txt.replace("\n", ""))
        results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)
    return results


def run_model_example(n_examples=5):
    global vocab_src, vocab_tgt, spacy_de, spacy_en

    print("Preparing Data ...")
    _, valid_dataloader = create_dataloaders(
        torch.device("cpu"),
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=1,
        is_distributed=False,
    )

    print("Loading Trained Model ...")

    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(
        torch.load("euro_final.pt", map_location=torch.device("cpu"))
    )

    print("Checking Model Outputs:")
    example_data = check_outputs(
        valid_dataloader, model, vocab_src, vocab_tgt, n_examples=n_examples
    )
    return model, example_data


if __name__ == "__main__":
    spacy_de, spacy_en, vocab_src, vocab_tgt = load_all_vocab()
    run_model_example()