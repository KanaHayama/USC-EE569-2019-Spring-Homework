import sys
import re

def main():
    train_acc = []
    test_acc = []
    for line in sys.stdin:
        train_match = re.findall(r" acc: (\d\.\d+)", line)
        test_match = re.findall(r"_acc: (\d\.\d+)", line)
        assert len(train_match) == len(test_match)
        if len(train_match) > 0 and len(test_match) > 0:
            train_acc.append(float(train_match[0]))
            test_acc.append(float(test_match[0]))
    
    print(train_acc)
    print(test_acc)

if __name__ == "__main__":
    main()