import functools

# 比較函數，按照長度進行比較
def compare_by_length(a, b):
    print(a,b,len(a) - len(b))
    return len(a) - len(b)

# 將比較函數轉換為鍵函數
key_func = functools.cmp_to_key(compare_by_length)

# 待排序的列表
words = ["a", "ab", "abc", "avcd"]

# 按照單詞長度排序
sorted_words = sorted(words, key=key_func)

# 輸出結果
print(sorted_words)  # ['date', 'apple', 'banana', 'cherry']
