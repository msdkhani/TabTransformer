
import torch
import torch.nn.functional as F


def return_int_vector(text, word_to_int, SEQUENCE_LENGTH):
    words = text.split()
    input_seq = torch.LongTensor([word_to_int[word] for word in words[-SEQUENCE_LENGTH:]]).unsqueeze(0)
    return input_seq


def sample_next(predictions):
    probabilities = F.softmax(predictions[:, -1, :], dim=-1).cpu()
    next_token = torch.argmax(probabilities)
    return int(next_token.cpu())


def text_generator_greedy(sentence, generate_length, SEQUENCE_LENGTH, model,word_to_int, int_to_word, device):

    sample = sentence
    for i in range(generate_length):
        int_vector = return_int_vector(sample, word_to_int, SEQUENCE_LENGTH)
        if len(int_vector) >= SEQUENCE_LENGTH - 1:
            break
        input_tensor = int_vector.to(device)
        with torch.no_grad():
            predictions = model(input_tensor)
        next_token = sample_next(predictions)
        sample += ' ' + int_to_word[next_token]
    print(sample)
    print('\n')


def sample_next_beam(predictions, top_k=3):
    probabilities = F.softmax(predictions[:, -1, :], dim=-1).cpu()
    top_probs, top_indices = torch.topk(probabilities, top_k)
    return top_probs, top_indices


def text_generator_beam(sentence, generate_length, SEQUENCE_LENGTH, model,word_to_int, int_to_word, device):

    sample = sentence
    for _ in range(generate_length):
        int_vector = return_int_vector(sample, word_to_int, SEQUENCE_LENGTH)
        input_tensor = int_vector.to(device)
        with torch.no_grad():
            predictions = model(input_tensor)
        top_probs, top_indices = sample_next_beam(predictions, top_k=5)
        print("Predicted next diseases with probabilities:")
        for i in range(top_indices.size(1)):
            disease = int_to_word[top_indices[0, i].item()]
            probability = top_probs[0, i].item()
            print(f"{disease} with probability {probability:.4f}")
        print('\n')
        sample += ' ' + int_to_word[top_indices[0, 0].item()]

    print("Generated sequence:", sample)
    print('\n')

