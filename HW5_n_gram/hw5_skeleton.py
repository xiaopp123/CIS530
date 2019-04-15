import math, random
import collections

################################################################################
# Part 0: Utility Functions
################################################################################

COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']

def start_pad(n):
    ''' Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return '~' * n

def ngrams(n, text):
    ''' Returns the ngrams of the text as tuples where the first element is
        the length-n context and the second is the character '''
    res = []
    for i in range(len(text)):
        if i < n:
            context = start_pad(n - i)
            context += text[:i]
        else:
            context = text[i - n : i]
        res.append(tuple([context, text[i]]))
    return res

def create_ngram_model(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the city names
        found in the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        model.update(f.read())
    return model

def create_ngram_model_lines(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the city names
        found in the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            model.update(line.strip())
    return model

################################################################################
# Part 1: Basic N-Gram Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''

    def __init__(self, n, k):
        self.n = n
        self.chars = collections.Counter()
        self.context2chars = {}
        self.k = k
        self.ngram = []

    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return set(self.chars.elements())

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        cur_ngram = ngrams(self.n, text)
        self.ngram.extend(cur_ngram)
        #print(self.ngram)

        for elt in cur_ngram:
            context = elt[0]
            char = elt[1]
            self.chars.update(char)
            self.context2chars.setdefault(context, collections.Counter())
            self.context2chars.get(context).update(char)

    def create(self, file_path):
        try:
            test = codecs.open(file_path, 'r', encoding='utf-8', errors='replace').read()
        except Exception as e:
            print(e)
        self.update(test)

    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        if context in self.context2chars.keys():
            context_num = sum(self.context2chars[context].values())
            if char in self.context2chars[context]:
                char_num = self.context2chars[context].get(char)
            else:
                char_num = 0

            char_smooth = char_num + self.k
            context_smooth = context_num + self.k * len(self.get_vocab())

            return char_smooth / context_smooth

        # if context is null
        return 1 / len(self.get_vocab())
            
    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''
        valid_chars = self.context2chars.get(context)
        r = random.random()
        psum = 0
        
        if valid_chars is not None and len(valid_chars) != 0:
            valid_chars = sorted(valid_chars)
            for i, char in enumerate(valid_chars):
                char_pro = self.prob(context, char)
                psum += char_pro
        #        print(i, context, char, r, psum)
                if psum >= r:
                    return char
        else:
            for i, char in enumerate(self.get_vocab()):
                char_pro = self.prob(context, char)
                psum += char_pro
        #        print(i, context, char, r, psum)
                if psum >= r:
                    return char

    def random_text(self, length):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''
        output = []
        context = collections.deque()

        for i in range(self.n):
            context.append('~')

        for i in range(length):
            char = self.random_char("".join(context))
            output.append(char)
            context.popleft()
            context.append(char)

        return "".join(output)

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        cur_ngram = ngrams(self.n, text)
        log_sum = 0
        for t in cur_ngram:
            pro = self.prob(t[0], t[1])
            if pro == 0:
                return math.inf
            log_sum += math.log(pro)

        return math.pow(math.e, log_sum / len(text) * -1.0)

################################################################################
# Part 2: N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, n, k):
        self.n = n
        self.chars = collections.Counter()
        self.context2chars = {}
        self.k = k
        self.ngram = []
        self.m = []
        self.lambdas = []
        for nz in range(self.n + 1):
            self.m.append(NgramModel(nz, self.k))
            self.lambdas.append(1 / (self.n + 1))

    def get_vocab(self):
        return self.m[0].get_vocab()

    def update(self, text):
        for mi in self.m:
            mi.update(text)
    def create(self, file_path):
        for mi in self.m:
             mi.create(file_path)

    def update_lambda(self, lambdas):
        if len(lambdas) == self.n:
            self.lambdas = lambdas
        else:
            print('length of lambdas is incorrect')

    def prob(self, context, char):
        prob_interp = 0
        #go through and get the smoothing probabilities of all the models, add them based on lambda weights
        for i, ms in enumerate(self.m):
            #print(ms.get_vocab())
            if i == 0:
                context_win = ''
            else:
                context_win = context[-i:]
            print(i, context, context_win)
            prob_interp = prob_interp + (self.lambdas[i] * ms.prob(context_win,char))

        return prob_interp
    
################################################################################
# Part 3: Your N-Gram Model Experimentation
################################################################################

if __name__ == '__main__':
    #s = 'abcdefghijk'
    #print(ngrams(3, s))
    #m = NgramModel(1, 0)
    #m.update('abab')
    #m.update('abcd')
    #random.seed(1)
    #print(m.random_text(25))
    #m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 7)
    #print("model created")
    #print(m.random_text(250))
    #m = NgramModel(1, 1)
    #m.update('abab')
    #m.update('abcd')
    #print(m.prob('a', 'a'))
    #print(m.prob('a', 'b'))
    #print(m.prob('c', 'd'))
    #print(m.prob('d', 'a'))
    #print(m.perplexity('abcd'))
    #print(m.perplexity('abca'))
    #print(m.perplexity('abcda'))

    #m = NgramModelWithInterpolation(2, 1)
    #m.update('abab')
    #m.update('abcd')
    #print(m.get_vocab())
    #print(m.prob('~a', 'b'))
    models = {city: NgramModelWithInterpolation(3, 1) for city in COUNTRY_CODES}
    for city, model in models:
        model.create()
    print(models)
