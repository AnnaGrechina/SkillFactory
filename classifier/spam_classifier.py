import math # импортировали для вычисления логарифма
import string  # для определения знаков препинания

# calculate_word_frequencies(self, rbody, label) : заполняет спам|не-спам словарь для 1го предложения
# train(self, train_data) : перебираем все предложения, отправляем каждое в calculate_word_frequencies
#                           и вычисляем общие значения (вероятность спама в мире, количество слов)
# calculate_P_Bi_A

class Classifier:
    def __init__(self):
        # создает экземпляр классификатора, которому передается обучающий набор данных
        # создает пустой словарь, в который будем записывать спам-слова:
        self.spam_words = dict()
        # создаем пустой словарь, в который будем записывать не-спам-слова:
        self.ham_words = dict()
        self.pA = 0.5 # вероятность встретить спам
        self.pNotA = 0.5 # вероятность встретить не-спам
        self.sum_spam_words = 0 # общее кол-во слов в спам текстах (неуникальное)
        self.sum_ham_words = 0  # общее кол-во слов в не-спам текстах (неуникальное)

        
        
    def calculate_word_frequencies(self, rbody, label):
        # функция актуализирует словари спам-слов и не-спам-слов для одного предложения из train_data. 
        # Ключ - слово, значение - сколько раз встретили
        body = self.text_clean(rbody) #приводим текст к нижнему регистру и убираем знаки прептнания
        for rword in body.split():
            word = self.words_only(rword)
            self.spam_words[word] = self.spam_words.get(word,0)
            self.ham_words[word] = self.ham_words.get(word,0)
            if label == 'SPAM': self.spam_words[word] += 1
            if label == 'NOT_SPAM': self.ham_words[word] += 1
        return

    
    def train(self, train_data):
	    # проходим по всему тренировочному набору и заполняем словари,
		# а также вычисляем частоту спам и неспам текстов
        self.train_data = train_data
        total_texts = 0
        spam_texts = 0
        ham_texts = 0
        for text, label in self.train_data:
            self.calculate_word_frequencies(text, label)
			# добавляем счетчик общего числа текстов и текстов с соответвующей меткой
            total_texts += 1
            if label == 'SPAM':
                spam_texts += 1
            elif label == 'NOT_SPAM':
                ham_texts += 1
		# после того, как прошлись по всему тренировочному набору,
		# вычисляем частоту спам-неспам, общее количество спам-слов и неспам-слов
        self.pA = spam_texts/total_texts # вероятность встретить спам
        self.pNotA = ham_texts/total_texts # вероятность встретить не-спам
        self.sum_spam_words = sum(self.spam_words.values())
        self.sum_ham_words = sum(self.ham_words.values())
        return
    
    def calculate_P_Bi_A(self, rword, label):
	# определяем для "тестового" предложения вероятность спамовости или неспамовости каждого слова
        word = self.words_only(rword)
        offset = 1
        if label == 'SPAM':
            res = (self.spam_words.get(word,0) + offset)/(self.sum_spam_words + offset * len(self.spam_words))
        elif label == 'NOT_SPAM':
            res = (self.ham_words.get(word,0) + offset)/(self.sum_ham_words + offset * len(self.ham_words))
        else:
            return None
        return res
    
    def calculate_log_P_B_A(self, text, label):
	# вычисляем логарифм вероятности, что текст является LABEL (спам или неспам)
	# вероятность "по умолчанию", на основе общего соотношения спам/неспам
        if label == 'SPAM':
            log_P_B_A = math.log(self.pA)
        elif label == 'NOT_SPAM':
            log_P_B_A = math.log(self.pNotA)
        else:
            return None
	# последовательно вычисляем вероятность для каждого слова и добавляем к результату
        for rword in text.split():
            word = self.words_only(rword)
            log_P_B_A += math.log(self.calculate_P_Bi_A(word, label))
        return log_P_B_A

    
    def text_clean(self, text):
        # приводит текст к нижнему регистру и удаляет знаки препинания из текста 
        return text.lower().translate(str.maketrans('', '', string.punctuation))
    
    def words_only(self, rword):
        # оставляем значащими только слова, а все последовательности цифр заменяем на "NUMBER"
        return rword if not rword.isnumeric() else 'NUMBER' 
    
 # основная функция для классификации выполнена с логарифмированием вероятности   
    def classify(self, remail):
        email = self.text_clean(remail)
        log_p_spam = self.calculate_log_P_B_A(email, 'SPAM')
        log_p_not_spam = self.calculate_log_P_B_A(email, 'NOT_SPAM')
        if log_p_spam > log_p_not_spam:
            return 'SPAM'
        else:
            return 'NOT_SPAM'
