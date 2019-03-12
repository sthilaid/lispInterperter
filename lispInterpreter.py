import sys
from enum import Enum
import unittest

LispLexerDebug = False

###############################################################################
## Lexing

class LexResult:
    def __init__(self, token, rest):
        self.token = token
        self.rest = rest

    def __repr__(self):
        return "LexResult(\"%s\", \"%s\")" % (self.token, self.rest)

def lexGeneric0(str, fn):
    if len(str) == 0:
        return False
    isLetter = fn(str[0])
    if isLetter:
        return LexResult(str[0], str[1:])
    else:
        return False

def lexCompose(str, fns, init=""):
    if LispLexerDebug : print("lexCompose(%s, %s, %s)" % (str, fns, init))
    if len(fns) == 0:
        return LexResult(init, str)
    else:
        res = fns[0](str)
        if res:
            comp = lexCompose(res.rest, fns[1:], init)
            if comp:
                return LexResult(res.token + comp.token, comp.rest)
            else:
                return False
        else:
            return False

def lexWord(str, word):
    if LispLexerDebug : print("lexWord(%s, %s)" % (str, word))
    wordLen = len(word)
    if str[0:wordLen] == word:
        return LexResult(word, str[wordLen:])
    else:
        return False

def lexMultiple(str, minCount, fn, init=""):
    if LispLexerDebug : print("lexMultiple(%s, %s, %s, %s)" % (str, minCount, fn, init))
    count = 0
    result = LexResult(init, str)
    
    while True:
        currentResult = fn(result.rest)
        # print("str: %s result: %s" % ((result.rest if result else False), result))
        if currentResult:
            count = count+1
            result.token    = result.token + currentResult.token
            result.rest     = currentResult.rest
            if not currentResult.rest:
                break #finished
        else:
            break

    # print("count: %d >= %d" % (count ,minCount))
    if count >= minCount:
        return result
    else:
        return False
    
def lexComment(str):
    if LispLexerDebug : print("lexComment(%s)" % str)
    length = len(str)
    if length == 0 or str[0] != ';':
        return False
    outStrIndex = length
    for i in range(length):
        if str[i] == '\n':
            outStrIndex = i+1
            break
    return LexResult(str[0:outStrIndex], str[outStrIndex:])

def lexWhitespace(str):
    if LispLexerDebug : print("lexWhitespace(%s)" % str)
    spaces = [' ', '\t', '\n', '\r']
    end = len(str)
    for i in range(end):
        if str[i] not in spaces:
            end = i
            break
    if end == 0:
        return False
    else:
        return LexResult(str[0:end], str[end:])

def lexDelimiter(str):
    if LispLexerDebug : print("lexDelimiter(%s)" % str)
    simpleDelmiters = ['|', '(', ')', '"', ';']
    return (lexGeneric0(str, lambda chr: chr in simpleDelmiters)
            or lexWhitespace(str))

def lexAtmosphere(str):
    if LispLexerDebug : print("lexAtmosphere(%s)" % str)
    return lexWhitespace(str) or lexComment(str)

def lexInterTokenSpace(str):
    if LispLexerDebug : print("lexInterTokenSpace(%s)" % str)
    return lexMultiple(str, 0, lexAtmosphere)

def lexLetter(str):
    if LispLexerDebug : print("lexLetter(%s)" % str)
    return lexGeneric0(str, lambda chr: ((chr >= 'a' and chr <= 'z')
                                         or (chr >= 'A' and chr <= 'Z')))

def lexSpecialInitial(str):
    if LispLexerDebug : print("lexSpecialInitial(%s)" % str)
    specialChars = ['!', '$', '%', '&', '*', '/', ':',
                    '<', '=', '>', '?', '^', '_', '-', '~']
    return lexGeneric0(str, lambda chr: chr in specialChars)

def lexDigit(str, base = 10):
    if LispLexerDebug : print("lexDigit(%s, %s)" % (str, base))
    if base == 16:
        return (lexGeneric0(str, lambda c: c >= '0' and c <= '9')
                or lexGeneric0(str, lambda c: c >= 'A' and c <= 'F')
                or lexGeneric0(str, lambda c: c >= 'a' and c <= 'f'))
    else:
        return lexGeneric0(str, lambda c: c >= '0' and c < chr(ord('0') + base))

def lexSpecialSubsequent(str):
    if LispLexerDebug : print("lexSubsequent(%s)" % str)
    chars = ['+', '-', '.', '@']
    return lexGeneric0(str, lambda chr: chr in chars)

def lexPeculiarIdentifier(str):
    if LispLexerDebug : print("lexPeculiarIdentifier(%s)" % str)
    chars = ['+', '-'] # missing '...' 
    return lexGeneric0(str, lambda chr: chr in chars)

def lexInitial(str):
    if LispLexerDebug : print("lexInitial(%s)" % str)
    return lexLetter(str) or lexSpecialInitial(str)

def lexSubsequent(str):
    if LispLexerDebug : print("lexSubsequent(%s)" % str)
    return lexInitial(str) or lexDigit(str) or lexSpecialSubsequent(str)

def lexIdentifier(str):
    if LispLexerDebug : print("lexIdentifier(%s)" % str)
    iniRes = lexInitial(str)
    if iniRes:
        subs = lexMultiple(iniRes.rest, 0, lexSubsequent)
        return (LexResult(iniRes.token + subs.token, subs.rest) if subs else False)
    else:
        return lexPeculiarIdentifier(str)

def lexExpressionKeyword(str):
    if LispLexerDebug : print("lexExpressionKeyword(%s)" % str)
    return (lexWord(str, "quote")
            or lexWord(str, "lambda")
            or lexWord(str, "if")
            or lexWord(str, "set!")
            or lexWord(str, "begin")
            or lexWord(str, "cond")
            or lexWord(str, "and")
            or lexWord(str, "or")
            or lexWord(str, "case")
            or lexWord(str, "let")
            or lexWord(str, "let*")
            or lexWord(str, "letrec")
            or lexWord(str, "do")
            or lexWord(str, "delay")
            or lexWord(str, "quasiquote"))

def lexSynacticKeyword(str):
    if LispLexerDebug : print("lexSynacticKeyword(%s)" % str)
    return (lexExpressionKeyword(str)
            or lexWord(str, "else")
            or lexWord(str, "=>")
            or lexWord(str, "define")
            or lexWord(str, "unquote")
            or lexWord(str, "unquote-splicing"))

def lexVariable(str):
    if LispLexerDebug : print("lexVariable(%s)" % str)
    if not lexSynacticKeyword(str):
        return lexIdentifier(str)
    else:
        return False

def lexBoolean(str):
    if LispLexerDebug : print("lexBoolean(%s)" % str)
    return lexWord(str, "#t") or lexWord(str, "#f")

def lexAnyCharacter(str):
    if LispLexerDebug : print("lexAnyCharacter(%s)" % str)
    return lexGeneric0(str, lambda x: True)

def lexSpecificCharacter(str, c):
    if LispLexerDebug : print("lexSpecificCharacter(%s, %s)" % (str, c))
    return lexGeneric0(str, lambda x: x == c)

def lexCharacterName(str):
    if LispLexerDebug : print("lexCharacterName(%s)" % str)
    return lexWord(str, "space") or lexWord(str, "newline")
    
def lexCharacter(str):
    if LispLexerDebug : print("lexCharacter(%s)" % str)
    init = lexWord(str, "#\\")
    if (init):
        res = lexCharacterName(init.rest) or lexAnyCharacter(init.rest)
        if res:
            return LexResult("#\\" + res.token, res.rest)
        else:
            return False
    else:
        return False

def lexStringElement(str):
    if LispLexerDebug : print("lexStringElement(%s)" % str)
    res =  lexWord(str, "\\\"") or lexWord(str, "\\\\")
    if res:
        return res
    else:
        any = lexAnyCharacter(str)
        if any.token != "\"" and any.token != "\\":
            return any
        else:
            return False

def lexString(str):
    if LispLexerDebug : print("lexString(%s)" % str)
    init = lexSpecificCharacter(str, '"')
    if init:
        content = lexMultiple(init.rest, 0, lexStringElement)
        if content and lexSpecificCharacter(content.rest, '"'):
            return LexResult("\""+content.token+"\"", content.rest[1:])
        else:
            return False
    else:
        return False

def lexEmpty(str):
    if LispLexerDebug : print("lexEmpty(%s)" % str)
    return LexResult("", str)

def lexSign(str):
    if LispLexerDebug : print("lexSign(%s)" % str)
    return (lexSpecificCharacter(str, "+")
            or lexSpecificCharacter(str, "-")
            or lexEmpty(str))

def lexUInteger(str, base):
    if LispLexerDebug : print("lexIdentifier(%s, %s)" % (str, base))
    return lexCompose(str, [lambda x: lexMultiple(x, 1, lambda y: lexDigit(y, base)),
                            lambda x: lexMultiple(x, 0, lambda y: lexSpecificCharacter(y, "#"))])

def lexRadix(str, base):
    if LispLexerDebug : print("lexRadix(%s, %s)" % (str, base))
    if base == 2:
        return lexWord(str, "#b")
    elif base == 8:
        return lexWord(str, "#o")
    elif base == 10:
        return lexWord(str, "#d") #or lexEmpty(str)
    elif base == 16:
        return lexWord(str, "#x")

def lexExponentMarker(str):
    if LispLexerDebug : print("lexExponentMarker(%s)" % (str))
    markers = "esfdl"
    return lexGeneric0(str, lambda chr: chr in markers)
    
def lexExactness(str):
    if LispLexerDebug : print("lexExactness(%s)" % (str))
    return lexWord(str, "#i") or lexWord(str, "#e") # or lexEmpty(str)
    
def lexNumberPrefix(str, base):
    if LispLexerDebug : print("lexNumberPrefix(%s, %s)" % (str, base))
    return (lexCompose(str, [lambda x: lexRadix(x, base), lexExactness])
            or lexCompose(str, [lexExactness, lambda x: lexRadix(x, base)])
            or lexRadix(str, base)
            # base 10 can omit the radix
            or (base == 10) and (lexExactness(str)
                                 or lexEmpty(str)))

def lexNumberSuffix(str):
    if LispLexerDebug : print("lexNumberSuffix(%s)" % (str))
    return (lexCompose(str, [lexExponentMarker, lexSign,
                             lambda x: lexMultiple(x, 1, lexDigit)])
            or lexEmpty(str))

def lexDecimal(str):
    if LispLexerDebug : print("lexDecimal(%s)" % (str))
    return (lexCompose(str, [lambda x: lexMultiple(x, 1, lexDigit),
                             lambda x: lexMultiple(x, 1, lambda y: lexSpecificCharacter(y, "#")),
                             lambda x: lexSpecificCharacter(x, "."),
                             lambda x: lexMultiple(x, 0, lambda y: lexSpecificCharacter(y, "#")),
                             lexNumberSuffix])
            or lexCompose(str, [lambda x: lexSpecificCharacter(x, "."),
                                lambda x: lexMultiple(x, 1, lexDigit),
                                lambda x: lexMultiple(x, 0, lambda y: lexSpecificCharacter(y, "#")),
                                lexNumberSuffix])
            or lexCompose(str, [lambda x: lexMultiple(x, 1, lexDigit),
                                lambda x: lexSpecificCharacter(x, "."),
                                lambda x: lexMultiple(x, 1, lexDigit),
                                lambda x: lexMultiple(x, 0, lambda y: lexSpecificCharacter(y, "#")),
                                lexNumberSuffix])
            or lexCompose(str, [lambda x: lexUInteger(x, 10),
                                lexNumberSuffix]))

def lexURealNumber(str, base):
    if LispLexerDebug : print("lexURealNumber(%s, %s)" % (str, base))
    return ((base == 10 and lexDecimal(str))
            or lexCompose(str, [lambda x: lexUInteger(x, base),
                                lambda x: lexSpecificCharacter(x, "/"),
                                lambda x: lexUInteger(x, base)])
            or lexUInteger(str, base))

def lexRealNumber(str, base):
    if LispLexerDebug : print("lexRealNumber(%s, %s)" % (str, base))
    return lexCompose(str, [lexSign, lambda x: lexURealNumber(x, base)])

def lexNumberComplex(str, base):
    if LispLexerDebug : print("lexNumberComplex(%s, %s)" % (str, base))
    return (lexCompose(str, [lambda x: lexRealNumber(x, base),
                             lambda x: lexSpecificCharacter(x, "@"),
                             lambda x: lexRealNumber(x, base)])
            or lexCompose(str, [lambda x: lexRealNumber(x, base),
                                lambda x: lexSpecificCharacter(x, "+"),
                                lambda x: lexURealNumber(x, base),
                                lambda x: lexSpecificCharacter(x, "i")])
            or lexCompose(str, [lambda x: lexRealNumber(x, base),
                                lambda x: lexSpecificCharacter(x, "-"),
                                lambda x: lexURealNumber(x, base),
                                lambda x: lexSpecificCharacter(x, "i")])
            or lexCompose(str, [lambda x: lexRealNumber(x, base),
                                lambda x: lexSpecificCharacter(x, "+"),
                                lambda x: lexSpecificCharacter(x, "i")])
            or lexCompose(str, [lambda x: lexRealNumber(x, base),
                                lambda x: lexSpecificCharacter(x, "-"),
                                lambda x: lexSpecificCharacter(x, "i")])
            or lexCompose(str, [lambda x: lexSpecificCharacter(x, "+"),
                                lambda x: lexURealNumber(x, base),
                                lambda x: lexSpecificCharacter(x, "i")])
            or lexCompose(str, [lambda x: lexSpecificCharacter(x, "-"),
                                lambda x: lexURealNumber(x, base),
                                lambda x: lexSpecificCharacter(x, "i")])
            or lexCompose(str, [lambda x: lexSpecificCharacter(x, "+"),
                                lambda x: lexSpecificCharacter(x, "i")])
            or lexCompose(str, [lambda x: lexSpecificCharacter(x, "-"),
                                lambda x: lexSpecificCharacter(x, "i")])
            or lexRealNumber(str, base))

def lexNumberBase(str, base):
    if LispLexerDebug : print("lexNumberBase(%s, %s)" % (str, base))
    return lexCompose(str, [lambda x: lexNumberPrefix(x, base),
                            lambda x: lexNumberComplex(x, base)])

def lexNumber(str):
    if LispLexerDebug : print("lexNumber(%s)" % (str))
    return (lexNumberBase(str, 2)
            or lexNumberBase(str, 8)
            or lexNumberBase(str, 10)
            or lexNumberBase(str, 16))

class LispTokenTypes(Enum):
    Identifier  = 0
    Boolean     = 1
    Number      = 2
    Character   = 3
    String      = 4
    LParen      = 5
    RParen      = 6
    SharpLParen = 7
    Quote       = 8
    BackQuote   = 9
    Comma       = 10
    CommaSplice = 11
    Dot         = 12
    
class LispToken:
    def __init__(self, tokenType, text):
        self.tokenType  = tokenType
        self.text       = text

    def __repr__(self):
        return "LispToken(%s, \"%s\")" % (self.tokenType, self.text)

def lexToken(str):
    if LispLexerDebug : print("lexToken(%s)" % (str))
    lex = lexIdentifier(str)
    if lex:
        return LexResult(LispToken(LispTokenTypes.Identifier, lex.token), lex.rest)

    lex = lexBoolean(str)
    if lex:
        return LexResult(LispToken(LispTokenTypes.Boolean, lex.token), lex.rest)

    lex = lexNumber(str)
    if lex:
        return LexResult(LispToken(LispTokenTypes.Number, lex.token), lex.rest)

    lex = lexCharacter(str)
    if lex:
        return LexResult(LispToken(LispTokenTypes.Character, lex.token), lex.rest)

    lex = lexString(str) 
    if lex:
        return LexResult(LispToken(LispTokenTypes.String, lex.token), lex.rest)

    lex = lexSpecificCharacter(str, "(")
    if lex:
        return LexResult(LispToken(LispTokenTypes.LParen, lex.token), lex.rest)

    lex = lexSpecificCharacter(str, ")")
    if lex:
        return LexResult(LispToken(LispTokenTypes.RParen, lex.token), lex.rest)

    lex = lexWord(str, "#(") 
    if lex:
        return LexResult(LispToken(LispTokenTypes.SharpLParen, lex.token), lex.rest)

    lex = lexSpecificCharacter(str, "'") 
    if lex:
        return LexResult(LispToken(LispTokenTypes.Quote, lex.token), lex.rest)

    lex = lexSpecificCharacter(str, "`")
    if lex:
        return LexResult(LispToken(LispTokenTypes.BackQuote, lex.token), lex.rest)

    lex = lexWord(str, ",@") 
    if lex:
        return LexResult(LispToken(LispTokenTypes.CommaSplice, lex.token), lex.rest)
    
    lex = lexSpecificCharacter(str, ",")
    if lex:
        return LexResult(LispToken(LispTokenTypes.Comma, lex.token), lex.rest)

    lex = lexSpecificCharacter(str, ".") 
    if lex:
        return LexResult(LispToken(LispTokenTypes.Dot, lex.token), lex.rest)
    
    return False # failed parsing...

def parseTokens(str):
    tokens = []
    delimiterTokens = [LispTokenTypes.Identifier, LispTokenTypes.Number,
                       LispTokenTypes.Character, LispTokenTypes.Dot]
    input = str
    while True:
        lex = lexToken(input)
        if not lex:
            if not input:
                return tokens
            else:
                return False
        else:
            input = lex.rest
            token = lex.token
            if token.tokenType in delimiterTokens:
                delim = lexDelimiter(input)
                if input and not delim :
                    return False
            rest = lexInterTokenSpace(input)
            if rest:
                input = rest.rest
            tokens += [token]
    return tokens

###############################################################################
## Datum Parsing (read)

class LispDatum:
    def __init__(self, tokens, rest):
        self.tokens = tokens
        self.rest = rest
    def __repr__(self):
        return "LispDatum(%s, \"%s\")" % (self.tokens, self.rest)

def parseTokenType(tokens, types):
    if not tokens or len(tokens) == 0 or not isinstance(tokens[0], LispToken):
        return False
    if tokens[0].tokenType in types:
        return LexResult([tokens[0]], tokens[1:])
    else:
        return False

def parseAbbrevPrefix(tokens):
    prefixTypes = [LispTokenTypes.Quote, LispTokenTypes.BackQuote,
                   LispTokenTypes.Comma, LispTokenTypes.CommaSplice]
    return parseTokenType(tokens, prefixTypes)

def parseSymbol(tokens):
    return parseTokenType(tokens, [LispTokenTypes.Identifier])

def parseSimpleDatum(tokens):
    simpleDatumTypes = [LispTokenTypes.Boolean, LispTokenTypes.Number,
                        LispTokenTypes.Character, LispTokenTypes.String]

    return parseTokenType(tokens, simpleDatumTypes) or parseSymbol(tokens)

def parseList(tokens):
    return (lexCompose(tokens, [lambda x: parseTokenType(x, [LispTokenTypes.LParen]),
                                lambda x: lexMultiple(x, 0, parseDatum, init=[]),
                                lambda x: parseTokenType(x, [LispTokenTypes.RParen])],
                       init=[])
            or lexCompose(tokens, [lambda x: parseTokenType(x, [LispTokenTypes.LParen]),
                                   lambda x: lexMultiple(x, 1, parseDatum, init=[]),
                                   lambda x: parseTokenType(x, [LispTokenTypes.Dot]),
                                   parseDatum,
                                   lambda x: parseTokenType(x, [LispTokenTypes.RParen])],
                          init=[])
            or lexCompose(tokens, [parseAbbrevPrefix, parseDatum], init=[]))

def parseVector(tokens):
    return lexCompose(tokens, [lambda x: parseTokenType(x, [LispTokenTypes.SharpLParen]),
                               lambda x: lexMultiple(x, 0, parseDatum, init=[]),
                               lambda x: parseTokenType(x, [LispTokenTypes.RParen])],
                      init=[])

def parseCompoundDatum(tokens):
    return parseList(tokens) or parseVector(tokens)

def parseDatum(tokens):
    return parseCompoundDatum(tokens) or parseSimpleDatum(tokens)

###############################################################################
## Expression Parsing



###############################################################################
## Unit tests

class TestLispLex(unittest.TestCase):
    def testComment(self):
        self.assertTrue(lexComment(";asdfasdf; asdsdaf").token == ";asdfasdf; asdsdaf")
        self.assertFalse(lexComment(" ;asdfasdf; asdsdaf"))
        self.assertFalse(lexComment("asdfsdf"))

    def testWhitespace(self):
        self.assertTrue(lexWhitespace("   \t  \n safadf").token == "   \t  \n ")
        self.assertTrue(lexWhitespace("\t  \n safadf").token == "\t  \n ")
        self.assertTrue(lexWhitespace("\n  \n safadf").token == "\n  \n ")
        self.assertTrue(lexWhitespace("\r  \n safadf").token == "\r  \n ")
        self.assertFalse(lexWhitespace("a  \n safadf"))

    def testDelimiter(self):
        self.assertTrue(lexDelimiter("|sdafsa|sdaf").token == "|")
        self.assertTrue(lexDelimiter("; comment").token == ";")
        self.assertTrue(lexDelimiter("   allo").token == "   ")
        self.assertTrue(lexDelimiter("\n(allo)").token == "\n")
        self.assertFalse(lexDelimiter("@llo)"))

    def testAtmosphere(self):
        self.assertTrue(lexAtmosphere("   allo").token == "   ")
        self.assertTrue(lexAtmosphere(";dsfffdfasf\n   allo").token == ";dsfffdfasf\n")
        self.assertTrue(lexAtmosphere("\n    \t;  dsfffdfasf\n   allo").token == "\n    \t")

    def testInterTokenSpace(self):
        self.assertTrue(lexInterTokenSpace(";dsfffdfasf\n   allo").token == ";dsfffdfasf\n   ")
        self.assertTrue(lexInterTokenSpace(";dsfffdfasf\n   \t\r  ;asdfsdaf\nallo").token == ";dsfffdfasf\n   \t\r  ;asdfsdaf\n")
        self.assertTrue(lexInterTokenSpace("\r;dsfffdfasfa\nallo   \tf\nallo").token == "\r;dsfffdfasfa\n")
        self.assertFalse(lexInterTokenSpace("#f\r;dsfffdfasfa\nallo   \tf\nallo").token == "\r;dsfffdfasfa\n")

    def testLetter(self):
        self.assertTrue(lexLetter("a"))
        self.assertTrue(lexLetter("z"))
        self.assertTrue(lexLetter("A"))
        self.assertTrue(lexLetter("Z"))
        self.assertFalse(lexLetter("("))
        self.assertFalse(lexLetter("7"))

    def testSpecialInitial(self):
        self.assertTrue(lexSpecialInitial("!"))
        self.assertTrue(lexSpecialInitial("*"))
        self.assertTrue(lexSpecialInitial("?"))
        self.assertFalse(lexSpecialInitial("a"))

    def testDigit(self):
        self.assertTrue(lexDigit("9"))
        self.assertTrue(lexDigit("1"))
        self.assertTrue(lexDigit("0"))
        self.assertFalse(lexDigit("a"))
        self.assertTrue(lexDigit("a", 16))
        self.assertTrue(lexDigit("D", 16))
        self.assertFalse(lexDigit("9", 8))

    def testSpecialSubsequent(self):
        self.assertTrue(lexSpecialSubsequent("@"))
        self.assertTrue(lexSpecialSubsequent("+"))
        self.assertTrue(lexSpecialSubsequent("-"))
        self.assertFalse(lexSpecialSubsequent("&"))
        self.assertFalse(lexSpecialSubsequent("a"))
        self.assertFalse(lexSpecialSubsequent("0"))

    def testPeculiarIdentifier(self):
        self.assertTrue(lexPeculiarIdentifier("+"))
        self.assertTrue(lexPeculiarIdentifier("-"))
        self.assertFalse(lexPeculiarIdentifier("@"))

    def testInitial(self):
        self.assertTrue(lexInitial("a"))
        self.assertTrue(lexInitial("*"))
        self.assertFalse(lexInitial("@"))

    def testSubsequent(self):
        self.assertTrue(lexSubsequent("a"))
        self.assertTrue(lexSubsequent("@"))
        self.assertTrue(lexSubsequent("5"))
        self.assertTrue(lexSubsequent("+"))
        self.assertFalse(lexSubsequent("\r"))

    def testIdentifier(self):
        self.assertTrue(lexIdentifier("allo*@556"))
        self.assertTrue(lexIdentifier("test!"))
        self.assertTrue(lexIdentifier("$test?"))
        self.assertTrue(lexIdentifier("<test>"))
        self.assertFalse(lexIdentifier("@allo"))
        self.assertFalse(lexIdentifier("#allo"))

    def testExpressionKeyword(self):
        self.assertTrue(lexExpressionKeyword("quote"))
        self.assertTrue(lexExpressionKeyword("lambda"))
        self.assertTrue(lexExpressionKeyword("if"))
        self.assertTrue(lexExpressionKeyword("cond"))
        self.assertFalse(lexExpressionKeyword("allo"))
        self.assertFalse(lexExpressionKeyword("quota"))

    def testSynacticKeyword(self):
        self.assertTrue(lexSynacticKeyword("else"))
        self.assertTrue(lexSynacticKeyword("define"))
        self.assertTrue(lexSynacticKeyword("unquote"))
        self.assertFalse(lexSynacticKeyword("1234"))
        self.assertFalse(lexSynacticKeyword("allo"))

    def testVariable(self):
        self.assertTrue(lexVariable("allo"))
        self.assertTrue(lexVariable("allo?"))
        self.assertTrue(lexVariable("<allo>?"))
        self.assertTrue(lexVariable("<define>?"))
        self.assertTrue(lexVariable("defin?"))
        self.assertFalse(lexVariable("define?"))
        self.assertFalse(lexVariable("if?"))

    def testBoolean(self):
        self.assertTrue(lexBoolean("#f"))
        self.assertTrue(lexBoolean("#t"))
        self.assertFalse(lexBoolean("true"))
        self.assertFalse(lexBoolean("false"))
        
    def testAnyCharacter(self):
        self.assertTrue(lexAnyCharacter("a"))
        self.assertTrue(lexAnyCharacter("#"))
        self.assertTrue(lexAnyCharacter("?"))
        self.assertTrue(lexAnyCharacter("("))
        
    def testCharacterName(self):
        self.assertTrue(lexCharacterName("space"))
        self.assertTrue(lexCharacterName("newline"))
        self.assertFalse(lexCharacterName("newlin"))
        self.assertFalse(lexCharacterName("alllo"))

    def testCharacter(self):
        self.assertTrue(lexCharacter("#\\a"))
        self.assertTrue(lexCharacter("#\\("))
        self.assertTrue(lexCharacter("#\\\\"))
        self.assertTrue(lexCharacter("#\\#"))
        self.assertFalse(lexCharacter("##"))

    def testString(self):
        self.assertTrue(lexString("\"asdflkj   sdlkfjf\""))
        self.assertTrue(lexString("\"asdflkj \\\"hello\\\"   sdlkfjf\""))
        self.assertTrue(lexString("\"asdflkj c:\\\\users\\\\john\""))
        self.assertFalse(lexString("allo"))
        self.assertFalse(lexString("\"allo"))
        self.assertFalse(lexString("allo\""))
        self.assertFalse(lexString("1234"))

    def testCompose(self):
        self.assertTrue(lexCompose(";asd\nallo  ifblabla", [lexComment, lexIdentifier, lexAtmosphere, lexExpressionKeyword]))

    # -- tests todo -- 
    def testSign(self):
        self.assertTrue(lexSign("+"))
        self.assertTrue(lexSign("-"))
        self.assertTrue(lexSign(""))

    def testUInteger(self):
        self.assertTrue(lexUInteger("12345", 10).token == "12345")
        self.assertTrue(lexUInteger("0989845", 10).token == "0989845")
        self.assertTrue(lexUInteger("09898453297237953795923", 10).token == "09898453297237953795923")
        self.assertTrue(lexUInteger("09898453297237953795923.111", 10).token == "09898453297237953795923")
        self.assertFalse(lexUInteger(".09898453297237953795923", 10))
        self.assertTrue(lexUInteger("01110101101", 2).token == "01110101101")
        self.assertFalse(lexUInteger("201110101101", 2))
        self.assertTrue(lexUInteger("156765137136734", 8).token == "156765137136734")
        self.assertTrue(lexUInteger("f123ABCDEF13", 16).token == "f123ABCDEF13")
        
    def testRadix(self):
        self.assertTrue(lexRadix("#b", 2))
        self.assertTrue(lexRadix("#o", 8))
        self.assertTrue(lexRadix("#d", 10))
        self.assertTrue(lexRadix("#x", 16))
        self.assertFalse(lexRadix("#x", 2))
        self.assertFalse(lexRadix("#o", 16))

    def testExponentMarker(self):
        self.assertTrue(lexExponentMarker("e"))
        self.assertTrue(lexExponentMarker("s"))
        self.assertTrue(lexExponentMarker("f"))
        self.assertTrue(lexExponentMarker("d"))
        self.assertTrue(lexExponentMarker("l"))
        self.assertFalse(lexExponentMarker("b"))

    def testExactness(self):
        self.assertTrue(lexExactness("#i"))
        self.assertTrue(lexExactness("#e"))

    def testNumberPrefix(self):
        self.assertTrue(lexNumberPrefix("#x#e", 16).token == "#x#e")
        self.assertTrue(lexNumberPrefix("#b#i", 2))
        self.assertTrue(lexNumberPrefix("#i#o", 8))
        self.assertTrue(lexNumberPrefix("#i#d", 10))
        self.assertTrue(lexNumberPrefix("#i", 10))
        self.assertTrue(lexNumberPrefix("", 10))
        
    def testNumberSuffix(self):
        self.assertTrue(lexNumberSuffix("e+12345allo").token == "e+12345")
        self.assertTrue(lexNumberSuffix("e-12345"))
        self.assertTrue(lexNumberSuffix("e12345"))
        self.assertTrue(lexNumberSuffix("s12345"))
        self.assertTrue(lexNumberSuffix("d+12345"))
        self.assertTrue(lexNumberSuffix("l+12345"))

    def testDecimal(self):
        self.assertTrue(lexDecimal("12345"))
        self.assertTrue(lexDecimal("12345e+10"))
        self.assertTrue(lexDecimal("12345e-0"))
        self.assertTrue(lexDecimal(".12345e-0"))
        self.assertFalse(lexDecimal(".e-0"))
        self.assertTrue(lexDecimal("12345.12345e10"))
        self.assertTrue(lexDecimal("12345.12345"))
        self.assertTrue(lexDecimal("12345###.###e10"))

    def testURealNumber(self):
        self.assertTrue(lexURealNumber("12345", 10))
        self.assertTrue(lexURealNumber("12345/12345", 10))
        self.assertTrue(lexURealNumber("123.456e-10", 10))
        self.assertTrue(lexURealNumber("11010101", 2))
        self.assertTrue(lexURealNumber("1214124677776", 8))
        self.assertTrue(lexURealNumber("a7d580fff/fcc445", 16))
        self.assertFalse(lexURealNumber("a7d580fff/fcc445", 8))
        self.assertTrue(lexURealNumber("123", 2).token != "123")

    def testRealNumber(self):
        self.assertTrue(lexRealNumber("+123.456e-10", 10).token == "+123.456e-10")
        self.assertTrue(lexRealNumber("-1234567", 8).token == "-1234567")
        self.assertTrue(lexRealNumber("+123456789abcdef", 16).token == "+123456789abcdef")
        self.assertTrue(lexRealNumber("+101010101111", 2).token == "+101010101111")
        self.assertTrue(lexRealNumber("+101210101111", 2).token != "+101210101111")

    def testNumberComplex(self):
        self.assertTrue(lexNumberComplex("123@456.789", 10).token == "123@456.789")
        self.assertTrue(lexNumberComplex("-123.456+789.10i", 10).token == "-123.456+789.10i")
        self.assertTrue(lexNumberComplex("-123.456-789.10e10i", 10).token == "-123.456-789.10e10i")
        self.assertTrue(lexNumberComplex("-123.456e10+i", 10).token == "-123.456e10+i")
        self.assertTrue(lexNumberComplex("-123.456e10-i", 10).token == "-123.456e10-i")
        self.assertTrue(lexNumberComplex("+i", 10).token == "+i")
        self.assertTrue(lexNumberComplex("+123.456e-10i", 10).token == "+123.456e-10i")
        self.assertTrue(lexNumberComplex("-123.456e-10i", 10).token == "-123.456e-10i")
        self.assertTrue(lexNumberComplex("10011-101100i", 2).token == "10011-101100i")
        self.assertTrue(lexNumberComplex("1234567+1234567i", 8).token == "1234567+1234567i")
        self.assertTrue(lexNumberComplex("-123456789abcdefi", 16).token == "-123456789abcdefi")

    def testNumberBase(self):
        self.assertTrue(lexNumberBase("#d1234.5678e-9+543.21i", 10).token == "#d1234.5678e-9+543.21i")
        self.assertTrue(lexNumberBase("#b10110", 2).token == "#b10110")
        self.assertTrue(lexNumberBase("#o1234567-i", 8).token == "#o1234567-i")
        self.assertTrue(lexNumberBase("#xabcdef-abc123i", 16).token == "#xabcdef-abc123i")
        self.assertTrue(lexNumberBase("#e#d1234.5678e-9+543.21i", 10).token == "#e#d1234.5678e-9+543.21i")

    def testNumber(self):
        self.assertTrue(lexNumber("#e1234.567e-89-i").token == "#e1234.567e-89-i");
        self.assertTrue(lexNumber("#i1234.567e-89-i").token == "#i1234.567e-89-i");
        self.assertTrue(lexNumber("-12.99e10i").token == "-12.99e10i");
        self.assertTrue(lexNumber("#b101011-011i").token == "#b101011-011i")
        self.assertTrue(lexNumber("#o#e1234/5676+77i").token == "#o#e1234/5676+77i")
        self.assertTrue(lexNumber("#xabcde/123-456abci").token == "#xabcde/123-456abci")
        self.assertTrue(lexNumber("#xabcde").token == "#xabcde")
        self.assertFalse(lexNumber("abcde"))

    def testToken(self):
        self.assertTrue(lexToken("allo"))
        self.assertTrue(lexToken("$hello"))
        self.assertTrue(lexToken("#t"))
        self.assertTrue(lexToken("#e#d123.456e-789/123.56-i"))
        self.assertTrue(lexToken("#\\a"))
        self.assertTrue(lexToken("\"hello \\\"world\\\"!\""))
        self.assertTrue(lexToken("("))
        self.assertTrue(lexToken(")"))
        self.assertTrue(lexToken("#("))
        self.assertTrue(lexToken("'"))
        self.assertTrue(lexToken("`"))
        self.assertTrue(lexToken(","))
        self.assertTrue(lexToken(",@"))
        self.assertTrue(lexToken("."))

    def testParseTokens(self):
        self.assertTrue(len(parseTokens("(allo 1.134e10+i #f #\\a #(allo \"world\"))")) == 10)
        
    def testParseTokenType(self):
        self.assertTrue(parseTokenType(parseTokens("1.23+5i"), [LispTokenTypes.Number]))
        self.assertTrue(parseTokenType(parseTokens("(allo)"), [LispTokenTypes.LParen]))
        self.assertTrue(parseTokenType(parseTokens(",@(allo)"), [LispTokenTypes.CommaSplice]))

    def testParseSimpleDatum(self):
        self.assertTrue(parseSimpleDatum(parseTokens("allo")).rest == [])
        self.assertTrue(parseSimpleDatum(parseTokens("1.13e-13+i")).rest == [])
        self.assertTrue(parseSimpleDatum(parseTokens("#f")).rest == [])
        self.assertTrue(parseSimpleDatum(parseTokens("\"stringg \\\"str\\\" hello\"")).rest == [])

    def testParseList(self):
        self.assertTrue(parseList(parseTokens("(allo (sdf . asfsadf))")).rest == [])
        self.assertTrue(parseList(parseTokens("(car . cdr)")).rest == [])
        self.assertTrue(parseList(parseTokens("'(car . cdr)")).rest == [])
        self.assertTrue(parseList(parseTokens("`(one two)")).rest == [])
        self.assertTrue(parseList(parseTokens(",(one two)")).rest == [])
        self.assertTrue(parseList(parseTokens(",@(one two)")).rest == [])

    def testParseVector(self):
        self.assertTrue(parseVector(parseTokens("#(1 2 3 4)")).rest == [])
        self.assertFalse(parseVector(parseTokens("#(1 2 3 . 4)")))
        self.assertFalse(parseVector(parseTokens("(1 2 3 4)")))

    def testParseCompoundDatum(self):
        self.assertTrue(parseCompoundDatum(parseTokens("'(allo #(1 2 3))")).rest == [])

    def testParseDatum(self):
        self.assertTrue(parseDatum(parseTokens(",@(allo ,(1 2 . nil) #(1 2 3))")).rest == [])
        self.assertFalse(parseDatum(parseTokens(",@(allo ,(1 2 . nil) #(1 2 . 3))")))

def runLispTests():
    unittest.TestLoader().loadTestsFromTestCase(TestLispLex).run(unittest.TextTestRunner(sys.stdout,True, 1).run(unittest.TestLoader().loadTestsFromTestCase(TestLispLex)))
