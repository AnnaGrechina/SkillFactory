import math # импортировали для вычисления логарифма
import string  # для определения знаков препинания

class Classifier:
    def __init__(self):
        # создает экземпляр классификатора, которому передается обучающий набор данных
        # создает пустой словарь, в который будем записывать спам-слова:
        self.spam_words = dict()
        # создаем пустой словарь, в который будем записывать не-спам-слова:
        self.ham_words = dict()
        self.pA = 0.5 # вероятность встретить спам
        self.pNotA = 0.5 # вероятность встретить не-спам
        self.number_unical_words = 0  # количество уникальных слов - заполним при обучении
        self.sum_spam_words = 0 # общее кол-во слов в спам текстах (неуникальное)
        self.sum_ham_words = 0  # общее кол-во слов в не-спам текстах (неуникальное)

        
        
    def calculate_word_frequencies(self, rbody, label):
        # функция заполняет словари спам-слов и не-спам-слов. 
        # Ключ - слово, значение - сколько раз встретили
        body = self.text_clean(rbody) #приводим текст к нижнему регистру и убираем знаки прептнания
        if label == 'SPAM':
            for rword in body.split():
                word = self.words_only(rword) #слова оставляем, последовательности цифр заменяем на 555
                if word not in self.spam_words: 
                    self.spam_words[word] = 0
                self.spam_words[word] += 1
        if label == 'NOT_SPAM':
            for rword in body.split():
                word = self.words_only(rword)
                if word not in self.ham_words: 
                    self.ham_words[word] = 0
                self.ham_words[word] += 1
        return

    
    def train(self, train_data):  
        self.train_data = train_data
        total_texts = 0
        spam_texts = 0
        ham_texts = 0
        for text in self.train_data:
            self.calculate_word_frequencies(*text) # вызываем функцию, которая заполняет словари
            total_texts += 1
            if text[1] == 'SPAM':
                spam_texts += 1
            elif text[1] == 'NOT_SPAM':
                ham_texts += 1
        self.pA = spam_texts/total_texts # вероятность встретить спам
        self.pNotA = ham_texts/total_texts # вероятность встретить не-спам
        self.number_unical_words = len(self.spam_words.keys()|self.ham_words.keys())
        self.sum_spam_words = sum(self.spam_words.values())
        self.sum_ham_words = sum(self.ham_words.values())
        return
    
    def calculate_P_Bi_A(self, rword, label):
        word = self.words_only(rword)
        offset = 1
        if label == 'SPAM':
		# вероятности со сглаживанием без нормирования, так получилось правильно 2 из 3 определить
            res = (self.spam_words.get(word,0) + offset)/(self.number_unical_words + self.sum_spam_words)
        elif label == 'NOT_SPAM':
		# вероятности со сглаживанием без нормирования, так получилось правильно 2 из 3 определить
            res = (self.ham_words.get(word,0) + offset)/(self.number_unical_words + self.sum_ham_words)
        else:
            return None
        return res   
    
    def calculate_log_P_B_A(self, text, label):
        if label == 'SPAM':
            log_P_B_A = math.log(self.pA)
        elif label == 'NOT_SPAM':
            log_P_B_A = math.log(self.pNotA)
        else:
            return None
        for rword in text.split():
            word = self.words_only(rword)
            log_P_B_A += math.log(self.calculate_P_Bi_A(word, label))
        return log_P_B_A

    
    def text_clean(self, text):
        # приводит текст к нижнему регистру и удалаяет знаки препинания из текста 
        return text.lower().translate(str.maketrans('', '', string.punctuation))
    
    def words_only(self, rword):
        # оставляем значащими только слова, а все последовательности цифр заменяем на "555"
        word = rword if not rword.isnumeric() else 'number'
        word = word if len(word) <20 else 'veryverylongword'
        return word
    
 # основная функция для классификации выполнена с логарифмированием вероятности   
    def classify(self, remail):
        email = self.text_clean(remail)
        log_p_spam = self.calculate_log_P_B_A(email, 'SPAM')
        log_p_not_spam = self.calculate_log_P_B_A(email, 'NOT_SPAM')
        if log_p_spam > log_p_not_spam:
            return 'SPAM'
        else:
            return 'NOT_SPAM'

