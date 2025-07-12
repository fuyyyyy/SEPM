import json
import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--annotation-file', type=str, default=None)
    parser.add_argument('--result-file', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--summary-output-dir', type=str, default=None)

    return parser.parse_args()


def calculate_accuracy(jsonl_file):
    correct_count = 0
    total_count = 0
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            
            category = data['question_id'].split('/')[0]
            correct_answer = data['text']

            options = {
                "A": "amusement",
                "B": "anger",
                "C": "awe",
                "D": "contentment",
                "E": "disgust",
                "F": "excitement",
                "G": "fear",
                "H": "sadness"
            }
            
            if options[correct_answer] == category: 
                correct_count += 1
            total_count += 1
    
    # 计算正确率
    accuracy = correct_count / total_count if total_count > 0 else 0
    return total_count, accuracy


args = get_args()
jsonl_file = args.result_file
total_count, accuracy = calculate_accuracy(jsonl_file)
print(f"Accuracy: {accuracy * 100:.2f}%")


if args.output_dir is not None:
    output_file = os.path.join(args.output_dir, 'result-emoset.txt')
    with open(output_file, 'w') as f:
        f.write('Samples: {}\nAccuracy on emoset: {:.2f}%\n'.format(total_count, accuracy * 100))


if args.summary_output_dir is not None: 
    with open(args.summary_output_dir, 'a') as f_sum:
        f_sum.write('\nSamples: {}\nAccuracy on emoset: {:.2f}%\n'.format(total_count, accuracy * 100))

