import sys
from enum import Enum
import unittest
import functools
import numpy as np

LispLexerDebug = False

###############################################################################
## AST definitions

class LispAST:
    def __init__(self):
        pass
    def eval(self, env):
        raise Exception("'eval' not implemented for type: "+type(self).__name__)

class LispAST_void(LispAST):
    def __init__(self, _, __, ___):
        pass

class LispAST_Real:
    def __init__(self, sign, numerator, denominator, text):
        self.sign           = sign
        self.numerator      = numerator
        self.denominator    = denominator
        self.text           = text
    def __repr__(self):
        return "LispAST_Real(%s,%s,%s,\"%s\")" % ( self.sign, self.numerator, self.denominator, self.text)
    def __str__(self):
        return str(self.text)

class LispAST_Number(LispAST):
    def __init__(self, radix=10, exactness=True, real=0, imag=0, text=""):
        self.radix = radix
        self.exactness = exactness
        self.real = real
        self.imag = imag
        self.text = text
    def __repr__(self):
        return "LispAST_Number(%s,%s,%s,%s,\"%s\")" % ( self.radix, self.exactness, self.real, self.imag, self.text)
    def __str__(self):
        return str(self.text)

    def eval(self, env):
        return LispEvalContext(env, LispValue(self, LispValueTypes.Number))

class LispAST_token(LispAST):
    def __init__(self, tokenType, text, extra = []):
        self.tokenType = tokenType
        self.text = text
        self.extra = extra
    def __repr__(self):
        return "LispAST_token(%s, \"%s\", %s)" % (self.tokenType, self.text, self.extra)
    def __str__(self):
        return self.text

    def eval(self, env):
        valueType = tokenTypeToValue(self.tokenType)
        return LispEvalContext(env, LispValue(self.extra, valueType))

class LispAST_Variable:
    def __init__(self, id):
        self.id = id
    def __repr__(self):
        return "LispAST_Variable(%s)" % ( self.id)
    def __str__(self):
        return "LispAST_Variable(%s)" % ( self.id)

    def eval(self, env):
        val = env.get(self.id)
        if val:
            return LispEvalContext(env, val)
        else:
            print("[LispEvalError]: Unknown variable: %s" % self.id)
            return LispEvalContext(env, LispValueFalse())

class LispAST_Datum(LispAST):
    def __init__(self, content):
        self.content = content
    def __repr__(self):
        return "LispAST_Datum(%s)" % self.content
    def __str__(self):
        return "%s" % map(str, self.content)

class LispAST_Symbol(LispAST):
    def __init__(self, _, txt, __):
        self.text = txt
    def __repr__(self):
        return "LispAST_Symbol(%s)" % self.text
    def __str__(self):
        return self.text

class LispAST_List(LispAST):
    def __init__(self, head, tail):
        self.head   = head
        self.tail   = tail
    def __repr__(self):
        return "LispAST_List(%s, %s)" % (self.head, self.tail)
    def __str__(self):
        if self.tail:
            return "(%s . %s)" % (list(map(str, self.head)), str(self.tail))
        else:
            return "(%s)" % list(map(str, self.head))

class LispAST_Vector(LispAST):
    def __init__(self, content):
        self.content = content
    def __repr__(self):
        return "LispAST_Vector(%s)" % self.content
    def __str__(self):
        return "#(%s)" % list(map(str, self.content))

class LispAST_Abbreviation(LispAST):
    def __init__(self, abbrev, datum):
        self.abbrev = abbrev
        self.datum  = datum
    def __repr__(self):
        return "LispAST_Abbreviation(%s, %s)" % (self.abbrev, self.datum)
    def __str__(self):
        return "%s%s" % (str(self.abbrev), str(self.datum))

class LispAST_Quotation(LispAST):
    def __init__(self, datum):
        self.datum  = datum
    def __repr__(self):
        return "LispAST_Quotation(%s)" % (self.datum)
    def __str__(self):
        return "'%s" % str(self.datum)

class LispAST_ProcedureCall(LispAST):
    def __init__(self, operator, operands):
        self.operator = operator
        self.operands = operands
    def __repr__(self):
        return "LispAST_ProcedureCall(%s, %s)" % (self.operator, self.operands)
    def __str__(self):
        return "(%s, %s)" % (self.operator, self.operands)

    def eval(self, env):
        fun     = self.operator.eval(env)
        if not fun or (fun.value.valueType != LispValueTypes.Procedure
                       and fun.value.valueType != LispValueTypes.Primitive):
            print("[LispEvalError] Invalid procedure call operator: %s" % self.operator)
            return LispValueFalse()

        args    = list(map(lambda x: x.eval(env).value, self.operands))
        funAst  = fun.value.value # context -> value -> proc

        if fun.value.valueType == LispValueTypes.Primitive:
            return funAst.apply(args)
            
        else: # LispValueTypes.Procedure
            newEnv  = fun.env.copy()
            if funAst.formals.varlist:
                newEnv[funAst.formals.varlist.id] = args
            else:
                hasRest     = funAst.formals.rest
                varsCount   = len(funAst.formals.vars)
                validArgNum = (len(args) >= varsCount) if hasRest else (len(args) == varsCount)
                if not validArgNum:
                    print("[LispEvalError] Invalid argment count for %s. Expecting %d, got %d."
                          % (self.operator, varsCount, len(args)))
                    return LispValueFalse()
                for i in range(varsCount):
                    newEnv[funAst.formals.vars[i].id] = args[i]
                if hasRest:
                    newEnv[funAst.formals.rest.id] = LispAST_List(args[varsCount:], False)
        
            bodyValue = funAst.body.eval(newEnv)
            return LispEvalContext(env, bodyValue.value)

class LispAST_Formals(LispAST):
    def __init__(self, varlist, vars, rest):
        self.varlist = varlist
        self.vars = vars
        self.rest = rest
    def __repr__(self):
        return "LispAST_Formals(%s,%s,%s)" % ( self.varlist, self.vars, self.rest)
    def __str__(self):
        return "LispAST_Formals(%s,%s,%s)" % ( self.varlist, self.vars, self.rest)

class LispAST_Body(LispAST):
    def __init__(self, definitions, body):
        self.definitions = definitions
        self.body = body
    def __repr__(self):
        return "LispAST_Body(%s,%s)" % ( self.definitions, self.body)
    def __str__(self):
        return "LispAST_Body(%s,%s)" % ( self.definitions, self.body)

    def eval(self, env):
        newEnv = env.copy()
        for i in range(len(self.definitions)):
            pass #todo: add definition to newEnv

        currentEnv  = env.copy()
        currentVal  = LispValueFalse()

        for b in range(len(self.body)):
            currentVal = self.body[b].eval(currentEnv)
            currentEnv = currentVal.env
        return LispEvalContext(currentEnv, currentVal)

class LispAST_LambdaExpression(LispAST):
    def __init__(self, formals, body):
        self.formals = formals
        self.body = body
    def __repr__(self):
        return "LispAST_LambdaExpression(%s,%s)" % ( self.formals, self.body)
    def __str__(self):
        return "LispAST_LambdaExpression(%s,%s)" % ( self.formals, self.body)

    def eval(self, env):
        return LispEvalContext(env, LispValue(self, LispValueTypes.Procedure, env.copy()))

class LispAST_Primitive(LispAST):
    def __init__(self, name, pyFun, argCount = False):
        self.name = name
        self.pyFun = pyFun
        self.argCount = argCount
    def __repr__(self):
        return "LispAST_Primitive(%s,%s)" % ( self.name, self.argCount)
    def __str__(self):
        return "LispAST_Primitive(%s,%s)" % ( self.name, self.argCount)

    def eval(self, env):
        return LispEvalContext(env, LispValue(self, LispValueTypes.Primitive))

    def apply(self, args):
        validArgNum = (len(args) == self.argCount) if self.argCount else True
        if not validArgNum:
            print("[LispEvalError] Invalid argment count for primitive %s. Expecting %d, got %d."
                  % (self.name, self.argCount, len(args)))
            return LispEvalContext([], LispValueFalse())
        else:
            return LispEvalContext([], self.pyFun(*args))

class LispAST_Conditional(LispAST):
    def __init__(self, test, consequent, alternate):
        self.test = test
        self.consequent = consequent
        self.alternate = alternate
    def __repr__(self):
        return "LispAST_Conditional(%s,%s,%s)" % ( self.test, self.consequent, self.alternate)
    def __str__(self):
        return "LispAST_Conditional(%s,%s,%s)" % ( self.test, self.consequent, self.alternate)

class LispAST_Assignement(LispAST):
    def __init__(self, var, exp):
        self.var = var
        self.exp = exp
    def __repr__(self):
        return "LispAST_Assignement(%s,%s)" % ( self.var, self.exp)
    def __str__(self):
        return "LispAST_Assignement(%s,%s)" % ( self.var, self.exp)

###############################################################################
## Lexing

class LexResult:
    def __init__(self, token, rest, extra=False):
        self.result = token
        self.rest = rest
        self.extra = extra
    def __repr__(self):
        return "LexResult(\"%s\", \"%s\", %s)" % (self.result, self.rest, self.extra)

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
                return LexResult(res.result + comp.result, comp.rest)
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
            result.result    = result.result + currentResult.result
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
        return (LexResult(iniRes.result + subs.result, subs.rest) if subs else False)
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

# def lexVariable(str):
#     if LispLexerDebug : print("lexVariable(%s)" % str)
#     if not lexSynacticKeyword(str):
#         return lexIdentifier(str)
#     else:
#         return False

def lexBoolean(str):
    if LispLexerDebug : print("lexBoolean(%s)" % str)
    if lexWord(str, "#t"):
        return LexResult("#t", str[2:], True)
    elif lexWord(str, "#f"):
        return LexResult("#f", str[2:], False)
    else:
        return False

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
            return LexResult("#\\" + res.result, res.rest, res.result)
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
        if any.result != "\"" and any.result != "\\":
            return any
        else:
            return False

def lexString(str):
    if LispLexerDebug : print("lexString(%s)" % str)
    init = lexSpecificCharacter(str, '"')
    if init:
        content = lexMultiple(init.rest, 0, lexStringElement)
        if content and lexSpecificCharacter(content.rest, '"'):
            return LexResult("\""+content.result+"\"", content.rest[1:], content.result)
        else:
            return False
    else:
        return False

def lexEmpty(str):
    if LispLexerDebug : print("lexEmpty(%s)" % str)
    return LexResult("", str)

###############################################################################
## Number lex / tokens

class container:
    def __init__(self, content=False):
        self.content = content

def assignContainer(res, varCont, fn = lambda x: x):
    if res and varCont:
        varCont.content = fn(res.extra)
    return res
    
def lexSign(str):
    if LispLexerDebug : print("lexSign(%s)" % str)
    return (lexSpecificCharacter(str, "+")
            or lexSpecificCharacter(str, "-")
            or lexEmpty(str))

def lexUInteger(str, base):
    if LispLexerDebug : print("lexUInteger(%s, %s)" % (str, base))
    result = lexCompose(str, [lambda x: lexMultiple(x, 1, lambda y: lexDigit(y, base)),
                              lambda x: lexMultiple(x, 0, lambda y: lexSpecificCharacter(y, "#"))])
    if result:
        result.extra = result.result
        return result
    else:
        return False

def lexRadix(str, base, numAST):
    if LispLexerDebug : print("lexRadix(%s, %s)" % (str, base))
    if base == 2:
        numAST.base = 2
        return lexWord(str, "#b")
    elif base == 8:
        numAST.base = 8
        return lexWord(str, "#o")
    elif base == 10:
        numAST.base = 10
        return lexWord(str, "#d") #or lexEmpty(str)
    elif base == 16:
        numAST.base = 16
        return lexWord(str, "#x")

def lexExponentMarker(str):
    if LispLexerDebug : print("lexExponentMarker(%s)" % (str))
    markers = "esfdl"
    return lexGeneric0(str, lambda chr: chr in markers)
    
def lexExactness(str, numAST):
    if LispLexerDebug : print("lexExactness(%s)" % (str))
    isExact = container(True)
    result = (assignContainer(lexWord(str, "#i"), isExact, lambda x: False)
              or assignContainer(lexWord(str, "#e"), isExact, lambda x: True)) # or lexEmpty(str)
    numAST.exactness = isExact.content
    return result
    
def lexNumberPrefix(str, base, numAST):
    if LispLexerDebug : print("lexNumberPrefix(%s, %s)" % (str, base))
    return (lexCompose(str, [lambda x: lexRadix(x, base, numAST),
                             lambda x: lexExactness(x, numAST)])
            or lexCompose(str, [lambda x: lexExactness(x, numAST),
                                lambda x: lexRadix(x, base, numAST)])
            or lexRadix(str, base, numAST)
            # base 10 can omit the radix
            or (base == 10) and (lexExactness(str, numAST)
                                 or lexEmpty(str)))

def lexNumberSuffix(str):
    if LispLexerDebug : print("lexNumberSuffix(%s)" % (str))
    return (lexCompose(str, [lexExponentMarker,
                             lexSign,
                             lambda x: lexMultiple(x, 1, lexDigit)])
            or lexEmpty(str))

def lexDecimal(str):
    if LispLexerDebug : print("lexDecimal(%s)" % (str))
    result = (lexCompose(str, [lambda x: lexMultiple(x, 1, lexDigit),
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
    if result:
        result.extra = result.result
        return result
    else:
        return False

def lexURealNumber(str, base):
    def toInt(n):
        if base == 10:
            # print("toInt(%s, %s): %d" % (n, base, int(float(n))))
            return int(float(n))
        else:
            # print("toInt(%s, %s): %d" % (n, base, int(n, base)))
            return int(n, base)

    def toFloat(n):
        # print("toFloat(%s): %.2f" %(n, float(n)))
        return float(n)

    if LispLexerDebug : print("lexURealNumber(%s, %s)" % (str, base))

    numerator   = container()
    denominator = container(1)
    
    result = ((base == 10 and assignContainer(lexDecimal(str), numerator, toFloat))
              or lexCompose(str, [lambda x: assignContainer(lexUInteger(x, base), numerator, toInt),
                                  lambda x: lexSpecificCharacter(x, "/"),
                                  lambda x: assignContainer(lexUInteger(x, base), denominator, toInt)])
              or assignContainer(lexUInteger(str, base), numerator, toInt))

    if result:
        result.extra = LispAST_Real(1, numerator.content, denominator.content, result.result)
        return result
    else:
        return False

def lexRealNumber(str, base):
    if LispLexerDebug : print("lexRealNumber(%s, %s)" % (str, base))

    sign = container(1)
    signResult = assignContainer(lexSign(str), sign, lambda s: -1 if s == "-" else 1)
    if not signResult:
        return False

    realResult = lexURealNumber(signResult.rest, base)
    if not realResult:
        return False

    realResult.result       = signResult.result + realResult.extra.text
    realResult.extra.text   = realResult.result
    realResult.extra.sign   = sign.content
    return realResult

def lexNumberComplex(str, base, numAST):
    def inverse(n):
        n.sign = -1 * n.sign
        return n

    if LispLexerDebug : print("lexNumberComplex(%s, %s)" % (str, base))

    real    = container(0)
    imag    = container(0)
    isPolar = container(False)

    result = (lexCompose(str, [lambda x: assignContainer(lexRealNumber(x, base), real),
                               lambda x: assignContainer(lexSpecificCharacter(x, "@"), isPolar),
                               lambda x: assignContainer(lexRealNumber(x, base), imag)])
              or lexCompose(str, [lambda x: assignContainer(lexRealNumber(x, base), real),
                                  lambda x: lexSpecificCharacter(x, "+"),
                                  lambda x: assignContainer(lexURealNumber(x, base), imag),
                                  lambda x: lexSpecificCharacter(x, "i")])
              or lexCompose(str, [lambda x: assignContainer(lexRealNumber(x, base), real),
                                  lambda x: lexSpecificCharacter(x, "-"),
                                  lambda x: assignContainer(lexURealNumber(x, base), imag, inverse),
                                  lambda x: lexSpecificCharacter(x, "i")])
              or lexCompose(str, [lambda x: assignContainer(lexRealNumber(x, base), real),
                                  lambda x: lexSpecificCharacter(x, "+"),
                                  lambda x: assignContainer(lexSpecificCharacter(x, "i"), imag, lambda y: LispAST_Real(1, 1, 1, ""))])
              or lexCompose(str, [lambda x: assignContainer(lexRealNumber(x, base), real),
                                  lambda x: lexSpecificCharacter(x, "-"),
                                  lambda x: assignContainer(lexSpecificCharacter(x, "i"), imag, lambda y: LispAST_Real(-1,1, 1, ""))])
              or lexCompose(str, [lambda x: lexSpecificCharacter(x, "+"),
                                  lambda x: lexURealNumber(x, base),
                                  lambda x: assignContainer(lexSpecificCharacter(x, "i"), imag)])
              or lexCompose(str, [lambda x: lexSpecificCharacter(x, "-"),
                                  lambda x: assignContainer(lexURealNumber(x, base), imag, lambda y: inverse),
                                  lambda x: lexSpecificCharacter(x, "i")])
              or lexCompose(str, [lambda x: lexSpecificCharacter(x, "+"),
                                  lambda x: assignContainer(lexSpecificCharacter(x, "i"), imag, lambda y: LispAST_Real(1, 1, 1, ""))])
              or lexCompose(str, [lambda x: lexSpecificCharacter(x, "-"),
                                  lambda x: assignContainer(lexSpecificCharacter(x, "i"), imag, lambda y: LispAST_Real(-1,1, 1, ""))])
              or assignContainer(lexRealNumber(str, base), real))

    if result:
        #todo polar form
        numAST.real = real.content
        numAST.imag = imag.content
        numAST.text = result.result
        return result
    else:
        return False

def lexNumberBase(str, base, numAST):
    if LispLexerDebug : print("lexNumberBase(%s, %s)" % (str, base))
    return lexCompose(str, [lambda x: lexNumberPrefix(x, base, numAST),
                            lambda x: lexNumberComplex(x, base, numAST)])

def lexNumber(str):
    if LispLexerDebug : print("lexNumber(%s)" % (str))
    numAST = LispAST_Number()
    result = (lexNumberBase(str, 2, numAST)
              or lexNumberBase(str, 8, numAST)
              or lexNumberBase(str, 10, numAST)
              or lexNumberBase(str, 16, numAST))
    if result:
        result.extra = numAST
        return result
    else:
        return False

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

def tokenTypeToValue(tokenType):
    if tokenType == LispTokenTypes.Boolean:
        return LispValueTypes.Boolean
    elif tokenType == LispTokenTypes.Character:
        return LispValueTypes.Char
    elif tokenType == LispTokenTypes.Number:
        return LispValueTypes.Number
    elif tokenType == LispTokenTypes.String:
        return LispValueTypes.String
    else:
        raise Exception("Token type: '%s' is not self evaluating...", self.tokenType)
    
class LispToken:
    def __init__(self, tokenType, text, extra=[]):
        self.tokenType  = tokenType
        self.text       = text
        self.extra      = extra

    def __repr__(self):
        return "LispToken(%s, \"%s\", %s)" % (self.tokenType, self.text, self.extra)

def lexToken(str):
    if LispLexerDebug : print("lexToken(%s)" % (str))
    lex = lexIdentifier(str)
    if lex:
        return LexResult(LispToken(LispTokenTypes.Identifier, lex.result), lex.rest)

    lex = lexBoolean(str)
    if lex:
        return LexResult(LispToken(LispTokenTypes.Boolean, lex.result, lex.extra), lex.rest)

    lex = lexNumber(str)
    if lex:
        return LexResult(LispToken(LispTokenTypes.Number, lex.result, lex.extra), lex.rest)

    lex = lexCharacter(str)
    if lex:
        return LexResult(LispToken(LispTokenTypes.Character, lex.result, lex.extra), lex.rest)

    lex = lexString(str) 
    if lex:
        return LexResult(LispToken(LispTokenTypes.String, lex.result, lex.extra), lex.rest)

    lex = lexSpecificCharacter(str, "(")
    if lex:
        return LexResult(LispToken(LispTokenTypes.LParen, lex.result), lex.rest)

    lex = lexSpecificCharacter(str, ")")
    if lex:
        return LexResult(LispToken(LispTokenTypes.RParen, lex.result), lex.rest)

    lex = lexWord(str, "#(") 
    if lex:
        return LexResult(LispToken(LispTokenTypes.SharpLParen, lex.result), lex.rest)

    lex = lexSpecificCharacter(str, "'") 
    if lex:
        return LexResult(LispToken(LispTokenTypes.Quote, lex.result), lex.rest)

    lex = lexSpecificCharacter(str, "`")
    if lex:
        return LexResult(LispToken(LispTokenTypes.BackQuote, lex.result), lex.rest)

    lex = lexWord(str, ",@") 
    if lex:
        return LexResult(LispToken(LispTokenTypes.CommaSplice, lex.result), lex.rest)
    
    lex = lexSpecificCharacter(str, ",")
    if lex:
        return LexResult(LispToken(LispTokenTypes.Comma, lex.result), lex.rest)

    lex = lexSpecificCharacter(str, ".") 
    if lex:
        return LexResult(LispToken(LispTokenTypes.Dot, lex.result), lex.rest)
    
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
            token = lex.result
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
## Evaluation and Values

class LispValueTypes(Enum):
    Boolean     = 0
    Symbol      = 1
    Char        = 2
    Vector      = 3
    Procedure   = 4
    Pair        = 5
    Number      = 6
    String      = 7
    Port        = 8
    Primitive   = 9

class LispValue:
    def __init__(self, value, valueType, env=False):
        self.value = value
        self.valueType = valueType
        self.env = env
    def __repr__(self):
        return "LispValue(%s,%s)" % ( self.value, self.valueType)
    def __str__(self):
        return "LispValue(%s,%s)" % ( self.value, self.valueType)

def LispValueFalse():
    return LispValue(LispAST_token(LispTokenTypes.Boolean, "#f", False), LispValueTypes.Boolean)

class LispEvalContext:
    def __init__(self, env, value):
        self.env = env
        self.value = value
    def __repr__(self):
        return "LispEvalContext(%s,%s)" % ( self.env, self.value)
    def __str__(self):
        return "LispEvalContext(%s,%s)" % ( self.env, self.value)

def lispEval(ast):
    primevalEnv = dict()
    primevalEnv['+'] = LispValue(LispAST_Primitive('+', lispPrimitive_add),
                                 LispValueTypes.Primitive)
    primevalEnv['-'] = LispValue(LispAST_Primitive('-', lispPrimitive_subtract),
                                 LispValueTypes.Primitive)
    return ast.eval(primevalEnv).value

def lispMakeNumberAST(num, denom=1, imagNum=0, imagDenom=1):
    sign = 1 if num >= 0 else -1
    realStr = "%.2f" % num if denom == 1 else "%.2f/%.2f" % (num, denom)
    imagStr = "" if imagNum == 0 else ("%.2f" % imagNum if imagDenom == 1 else "%.2f/%.2f" % (imagNum, imagDenom))
    numStr = "%s" % realStr if imagStr == "" else "%s + %si" % (realStr, imagStr)
    return LispValue(LispAST_Number(10, True,
                                    LispAST_Real(sign, num, denom, realStr),
                                    LispAST_Real(1, imagNum, imagDenom, imagStr),
                                    numStr),
                     LispValueTypes.Number)

def lispApplyNumberFun(x, y, f):
    if x.valueType != LispValueTypes.Number or y.valueType != LispValueTypes.Number:
        print("[LispEvalError] Invalid argment to addition primitive. Expecting Number/Number and got %s/%s" % (x.valueType, y.valueType))
        return LispEvalContext([], LispValueFalse())

    realNum, realDenom = f(x.value.real, y.value.real)
    imagNum, imagDenom = f(x.value.imag, y.value.imag)
    return lispMakeNumberAST(realNum, realDenom, imagNum, imagDenom)


def lispPrimitive_add(*numbers):
    def add(x, y):
        lcd = np.lcm(x.denominator, y.denominator)
        numX = x.numerator * lcd / x.denominator
        numY = y.numerator * lcd / y.denominator
        return (numX+numY, lcd)

    result = functools.reduce(lambda a,x: lispApplyNumberFun(a, x, add),
                              numbers,
                              lispMakeNumberAST(0))
    return result

def lispPrimitive_subtract(*numbers):
    def sub(x, y):
        lcd = np.lcm(x.denominator, y.denominator)
        numX = x.numerator * lcd / x.denominator
        numY = y.numerator * lcd / y.denominator
        return (numX-numY, lcd)

    if (len(numbers) == 1):
        return lispMakeNumberAST(-numbers[0].value.real.numerator,
                                 numbers[0].value.real.denominator,
                                 -numbers[0].value.imag.numerator,
                                 numbers[0].value.imag.denominator)
    elif (len(numbers) < 2):
        print("[LispEvalError] Invalid argment count for '-'. Expecting at least 1, got %d" % len(numbers))
        return LispEvalContext([], LispValueFalse())

    result = functools.reduce(lambda a,x: lispApplyNumberFun(a, x, sub),
                              numbers[1:],
                              numbers[0])
    return result

###############################################################################
## Parsing

class ParseResult:
    def __init__(self, ast, rest):
        self.result = [ast]
        self.rest = rest

    def __repr__(self):
        return "ParseResult(%s, %s)" % (self.result, self.rest)

def parseTokenType(tokens, tokenType, astType):
    if not tokens or len(tokens) == 0 or not isinstance(tokens[0], LispToken):
        return False
    if tokens[0].tokenType == tokenType:
        return ParseResult(astType(tokens[0].tokenType, tokens[0].text, tokens[0].extra), tokens[1:])
    else:
        return False

def parseTokenTypes(tokens, types, astType):
    if not tokens or len(tokens) == 0 or not isinstance(tokens[0], LispToken):
        return False
    if tokens[0].tokenType in types:
        return ParseResult(astType(tokens[0].tokenType, tokens[0].text, tokens[0].extra), tokens[1:])
    else:
        return False

def parseNumber(tokens):
    if not tokens or len(tokens) == 0 or not isinstance(tokens[0], LispToken):
        return False
    if tokens[0].tokenType == LispTokenTypes.Number:
        return ParseResult(tokens[0].extra, tokens[1:])
    else:
        return False

def cleanUpAsts(asts):
    def clean(a, x):
        if isinstance(x, LispAST_void):
            return a
        else:
            return a + [x]

    return functools.reduce(clean, asts, [])

def parseCompose(tokens, compFns, astType):
    if not tokens or (len(tokens) == 0 and len(compFns) > 0):
        return False

    result = lexCompose(tokens, compFns, init=[])
    if result:
        asts = result.result
        cleanAsts = cleanUpAsts(asts)
        # print("test: %s" % cleanAsts)
        return ParseResult(astType(*cleanAsts), result.rest)
    else:
        return False

def parseMultiple(tokens, n, fn):
    lexResult = lexMultiple(tokens, n, fn, init=[])
    if lexResult:
        asts = lexResult.result
        cleanAsts = cleanUpAsts(asts)
        return ParseResult(cleanAsts, lexResult.rest)
    else:
        return False

def parseSpecificIdentifier(tokens, id):
    identifier = parseTokenType(tokens, LispTokenTypes.Identifier, LispAST_token)
    if not identifier or identifier.result[0].text != id:
        return False
    else:
        return identifier

def parseSExp(tokens, id, bodyParseList, astType, startToken=LispTokenTypes.LParen, endToken=LispTokenTypes.RParen):
    resOffset = 1
    if not parseTokenType(tokens, startToken, LispAST_token):
        return False

    if id != "":
        if not parseSpecificIdentifier(tokens[1:], id):
            return False
        else:
            resOffset = 2

    result = parseCompose(tokens[resOffset:], bodyParseList, astType)

    if not result:
        return False
    if parseTokenType(result.rest, endToken, LispAST_token):
        result.rest = result.rest[1:]
        return result
    else:
        return False

###############################################################################
## Datum Parsing (read)

def parseAbbrevPrefix(tokens):
    prefixTypes = [LispTokenTypes.Quote, LispTokenTypes.BackQuote,
                   LispTokenTypes.Comma, LispTokenTypes.CommaSplice]
    return parseTokenTypes(tokens, prefixTypes, LispAST_token)

def parseSymbol(tokens):
    return parseTokenType(tokens, LispTokenTypes.Identifier, LispAST_Symbol)

def parseSimpleDatum(tokens):
    simpleDatumTypes = [LispTokenTypes.Boolean, LispTokenTypes.Number,
                        LispTokenTypes.Character, LispTokenTypes.String]

    return parseTokenTypes(tokens, simpleDatumTypes, LispAST_token) or parseSymbol(tokens)

def parseList(tokens):
    return (parseSExp(tokens, "",
                      [lambda x: parseMultiple(x, 0, parseDatum)],
                      lambda x: LispAST_List(x, False))
            or parseSExp(tokens, "",
                         [lambda x: parseMultiple(x, 0, parseDatum),
                          lambda x: parseTokenType(x, LispTokenTypes.Dot, LispAST_void),
                          parseDatum],
                         LispAST_List))
def parseVector(tokens):
    return parseSExp(tokens, "",
                     [lambda x: parseMultiple(x, 0, parseDatum)],
                     LispAST_Vector,
                     startToken=LispTokenTypes.SharpLParen,
                     endToken=LispTokenTypes.RParen)

def parseAbbreviation(tokens):
    return parseCompose(tokens, [parseAbbrevPrefix, parseDatum], LispAST_Abbreviation)

def parseCompoundDatum(tokens):
    return parseList(tokens) or parseVector(tokens) or parseAbbreviation(tokens)

def parseDatum(tokens):
    return parseCompoundDatum(tokens) or parseSimpleDatum(tokens)

###############################################################################
## Expression Parsing

def parseQuotation(tokens):
    return (parseCompose(tokens, [lambda x: parseTokenType(x, LispTokenTypes.Quote, LispAST_void),
                                  parseDatum],
                         LispAST_Quotation)
            or parseSExp(tokens, "quote", [parseDatum], LispAST_Quotation))

def parseSelfEvaluating(tokens):
    return (parseTokenTypes(tokens, [LispTokenTypes.Boolean,
                                     LispTokenTypes.Character, LispTokenTypes.String],
                            LispAST_token)
            or parseNumber(tokens))

def parseLitteral(tokens):
    if LispLexerDebug : print("parseLitteral(%s)" % tokens)
    return parseQuotation(tokens) or parseSelfEvaluating(tokens)

def parseVariable(tokens):
    if LispLexerDebug : print("parseVariable(%s)" % tokens)
    id = parseTokenType(tokens, LispTokenTypes.Identifier, LispAST_token)
    if id and not lexSynacticKeyword(id.result[0].text):
        return ParseResult(LispAST_Variable(id.result[0].text), id.rest)
    else:
        return False

def parseOperator(tokens):
    return parseExpression(tokens)

def parseOperands(tokens):
    return parseMultiple(tokens, 0, parseExpression)

def parseProcedureCall(tokens):
    if LispLexerDebug : print("parseProcedureCall(%s)" % tokens)
    return parseSExp(tokens, "", [parseOperator, parseOperands], LispAST_ProcedureCall)

def parseFormals(tokens):
    if LispLexerDebug : print("parseFormals(%s)" % tokens)
    return (parseSExp(tokens, "", [lambda x: parseMultiple(x, 0, parseVariable)],
                      lambda x: LispAST_Formals(False, x, False))
            or parseSExp(tokens, "", [lambda x: parseMultiple(x, 0, parseVariable),
                                      lambda x: parseTokenType(x, LispTokenTypes.Dot, LispAST_void),
                                      parseVariable],
                         lambda x,y: LispAST_Formals(False, x, y))
            or (lambda x: LispAST_Formals(x.result[0], False, False) if x else False)(parseVariable(tokens)))

def parseCommand(tokens):
    return parseExpression(tokens);

def parseSequence(tokens):
    if LispLexerDebug : print("parseSequence(%s)" % tokens)
    return parseMultiple(tokens, 1, parseCommand)

def parseBody(tokens):
    if LispLexerDebug : print("parseBody(%s)" % tokens)
    return parseCompose(tokens, [lambda x: parseMultiple(x, 0, parseDefinition),
                                 parseSequence],
                        LispAST_Body)

def parseLambdaExpression(tokens):
    if LispLexerDebug : print("parseLambdaExpression(%s)" % tokens)
    return parseSExp(tokens, "lambda", [parseFormals, parseBody], LispAST_LambdaExpression)

def parseTest(tokens):
    if LispLexerDebug : print("parseTest(%s)" % tokens)
    return parseExpression(tokens)

def parseConsequent(tokens):
    if LispLexerDebug : print("parseConsequent(%s)" % tokens)
    return parseExpression(tokens)

def parseAlternate(tokens):
    if LispLexerDebug : print("parseAlternate(%s)" % tokens)
    return parseExpression(tokens) or ParseResult([False], tokens) # or empty

def parseConditional(tokens):
    if LispLexerDebug : print("parseConditional(%s)" % tokens)
    return parseSExp(tokens, "if", [parseTest, parseConsequent, parseAlternate], LispAST_Conditional)

def parseAssignement(tokens):
    if LispLexerDebug : print("parseAssignement(%s)" % tokens)
    return parseSExp(tokens, "set!", [parseVariable, parseExpression], LispAST_Assignement)

# def parseCondClause(tokens):
#     return (parseSExp(tokens, [parseTest, parseSequence])
#             or parseSExp(tokens, [parseTest])
#             or parseSExp(tokens, [parseTest,
#                                   lambda x: parseSpecificIdentifier(x, "=>"),
#                                   parseExpression]))

# def parseCaseClause(tokens):
#     return parseSExp(tokens, [lambda x: parseSExp(x, [lambda y: lexMultiple(y,
#                                                                             0,
#                                                                             parseDatum,
#                                                                             init=[])]),
#                               parseSequence])

# def parseBindingSpec(tokens):
#     return parseSExp(tokens, [parseVariable, parseExpression])

# def parseIterationSpec(tokens):
#     return (parseSExp(tokens, [parseVariable, parseExpression, parseExpression])
#             or parseSExp(tokens, [parseVariable, parseExpression]))

# def parseDoResult(tokens):
#     return parseSequence(tokens) or LexResult([], tokens)

# def parseKeyword(tokens):
#     return parseTokenType(tokens, [LispTokenTypes.Identifier])

def parseDerivedExpression(tokens):
    return False
#     if LispLexerDebug : print("parseDerivedExpression(%s)" % tokens)
#     return (parseSExpWithId(tokens, "cond",     [lambda x: lexMultiple(x, 0, parseCondClause, init=[]),
#                                                  lambda x: parseSExpWithId(x, "else", [parseSequence])])
#             or parseSExpWithId(tokens, "cond",  [lambda x: lexMultiple(x, 1, parseCondClause, init=[])])
#             or parseSExpWithId(tokens, "case",  [parseExpression,
#                                                  lambda x: lexMultiple(x, 0, parseCaseClause, init=[]),
#                                                  lambda x: parseSExpWithId(x, "else", [parseSequence])])
#             or parseSExpWithId(tokens, "case",  [parseExpression, lambda x: lexMultiple(x, 1, parseCaseClause, init=[])])
#             or parseSExpWithId(tokens, "and",   [lambda x: lexMultiple(x, 0, parseTest, init=[])])
#             or parseSExpWithId(tokens, "or",    [lambda x: lexMultiple(x, 0, parseTest, init=[])])
#             or parseSExpWithId(tokens, "let",   [lambda x: parseSExp(x, [lambda y: lexMultiple(y, 0, parseBindingSpec, init=[])]),
#                                                  parseBody])
#             or parseSExpWithId(tokens, "let",   [parseVariable,
#                                                  lambda x: parseSExp(x, [lambda y: lexMultiple(y, 0, parseBindingSpec, init=[])]),
#                                                  parseBody])
#             or parseSExpWithId(tokens, "let*",  [lambda x: parseSExp(x, [lambda y: lexMultiple(y, 0, parseBindingSpec, init=[])]),
#                                                  parseBody])
#             or parseSExpWithId(tokens, "letrec",[lambda x: parseSExp(x, [lambda y: lexMultiple(y, 0, parseBindingSpec, init=[])]),
#                                                  parseBody])
#             or parseSExpWithId(tokens, "begin", [parseSequence])
#             or parseSExpWithId(tokens, "do",    [lambda x: parseSExp(x, [lambda y: lexMultiple(y, 0, parseIterationSpec, init=[])]),
#                                                  lambda x: parseSExp(x, [parseTest, parseDoResult]),
#                                                  lambda x: lexMultiple(x, 0, parseExpression, init=[])])
#             or parseSExpWithId(tokens, "delay", [parseExpression])
#             or parseQuasiQuotation(tokens))

def parseMacroUse(tokens):
    return False
#     if LispLexerDebug : print("parseMacroUse(%s)" % tokens)
#     return parseSExp(tokens, [parseKeyword,
#                               lambda x: lexMultiple(x, 0, parseDatum, init=[])])
def parseMacroBlock(tokens):
    return False
#     if LispLexerDebug : print("parseMacroBlock(%s)" % tokens)
#     return (parseSExpWithId(tokens, "let-syntax",
#                             [lambda x: parseSExp(x, [lambda y: lexMultiple(y,
#                                                                            0,
#                                                                            parseSyntaxSpec,
#                                                                            init=[])]),
#                              parseBody])
#             or parseSExpWithId(tokens, "letrec-syntax",
#                             [lambda x: parseSExp(x, [lambda y: lexMultiple(y,
#                                                                            0,
#                                                                            parseSyntaxSpec,
#                                                                            init=[])]),
#                              parseBody]))

def parseExpression(tokens):
    if LispLexerDebug : print("parseExpression(%s)" % tokens)
    return (parseVariable(tokens)
            or parseLitteral(tokens)
            or parseProcedureCall(tokens)
            or parseLambdaExpression(tokens)
            or parseConditional(tokens)
            or parseAssignement(tokens)
            or parseDerivedExpression(tokens)
            or parseMacroUse(tokens)
            or parseMacroBlock(tokens))

# def parseDefFormals(tokens):
#     return (lexMultiple(tokens, 0, parseVariable, init=[])
#             or lexCompose(tokens, [lambda x: lexMultiple(x, 0, parseVariable),
#                                    lambda x: parseSpecificIdentifier(x, LispTokenTypes.Dot),
#                                    parseVariable]))
    
def parseDefinition(tokens):
    return False
#     if LispLexerDebug : print("parseDefinition(%s)" % tokens)
#     return (parseSExpWithId(tokens, "define", [parseVariable, parseExpression])
#             or parseSExpWithId(tokens, "define", [lambda x: parseSExp(x, [parseVariable, parseDefFormals]),
#                                                   parseBody])
#             or parseSExpWithId(tokens, "begin", [lambda x: lexMultiple(x, 0, parseDefinition)]))

# def parseSyntaxDefinition(tokens):
#     if LispLexerDebug : print("parseSyntaxDefinition(%s)" % tokens)
#     return parseSExpWithId(tokens, "define-syntax", [parseKeyword, parseTransformerSpec])


# def parseCommandOrDefinition(tokens):
#     return (parseCommand(tokens)
#             or parseDefinition(tokens)
#             or parseSyntaxDefinition(tokens)
#             or parseSExpWithId(tokens, "begin", [lambda x: lexMultiple(x, 1, [parseCommandOrDefinition], init=[])]))
    
# def parseProgram(tokens):
#     if LispLexerDebug : print("parseProgram(%s)" % tokens)
#     return parseCommandOrDefinition(tokens)

# def parseSyntaxSpec(tokens):
#     # return parseSExp(tokens, [parseKeyword, parseTransformerSpec])
#     return False # todo...

# def parseQuasiQuotation(tokens):
#     if LispLexerDebug : print("parseQuasiQuotation(%s)" % tokens)
#     return False #todo


###############################################################################
## Unit tests

class TestLispLex(unittest.TestCase):
    def testComment(self):
        self.assertTrue(lexComment(";asdfasdf; asdsdaf").result == ";asdfasdf; asdsdaf")
        self.assertFalse(lexComment(" ;asdfasdf; asdsdaf"))
        self.assertFalse(lexComment("asdfsdf"))

    def testWhitespace(self):
        self.assertTrue(lexWhitespace("   \t  \n safadf").result == "   \t  \n ")
        self.assertTrue(lexWhitespace("\t  \n safadf").result == "\t  \n ")
        self.assertTrue(lexWhitespace("\n  \n safadf").result == "\n  \n ")
        self.assertTrue(lexWhitespace("\r  \n safadf").result == "\r  \n ")
        self.assertFalse(lexWhitespace("a  \n safadf"))

    def testDelimiter(self):
        self.assertTrue(lexDelimiter("|sdafsa|sdaf").result == "|")
        self.assertTrue(lexDelimiter("; comment").result == ";")
        self.assertTrue(lexDelimiter("   allo").result == "   ")
        self.assertTrue(lexDelimiter("\n(allo)").result == "\n")
        self.assertFalse(lexDelimiter("@llo)"))

    def testAtmosphere(self):
        self.assertTrue(lexAtmosphere("   allo").result == "   ")
        self.assertTrue(lexAtmosphere(";dsfffdfasf\n   allo").result == ";dsfffdfasf\n")
        self.assertTrue(lexAtmosphere("\n    \t;  dsfffdfasf\n   allo").result == "\n    \t")

    def testInterTokenSpace(self):
        self.assertTrue(lexInterTokenSpace(";dsfffdfasf\n   allo").result == ";dsfffdfasf\n   ")
        self.assertTrue(lexInterTokenSpace(";dsfffdfasf\n   \t\r  ;asdfsdaf\nallo").result == ";dsfffdfasf\n   \t\r  ;asdfsdaf\n")
        self.assertTrue(lexInterTokenSpace("\r;dsfffdfasfa\nallo   \tf\nallo").result == "\r;dsfffdfasfa\n")
        self.assertFalse(lexInterTokenSpace("#f\r;dsfffdfasfa\nallo   \tf\nallo").result == "\r;dsfffdfasfa\n")

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

    def testSign(self):
        self.assertTrue(lexSign("+"))
        self.assertTrue(lexSign("-"))
        self.assertTrue(lexSign(""))

    def testUInteger(self):
        self.assertTrue(lexUInteger("12345", 10).result == "12345")
        self.assertTrue(lexUInteger("0989845", 10).result == "0989845")
        self.assertTrue(lexUInteger("09898453297237953795923", 10).result == "09898453297237953795923")
        self.assertTrue(lexUInteger("09898453297237953795923.111", 10).result == "09898453297237953795923")
        self.assertFalse(lexUInteger(".09898453297237953795923", 10))
        self.assertTrue(lexUInteger("01110101101", 2).result == "01110101101")
        self.assertFalse(lexUInteger("201110101101", 2))
        self.assertTrue(lexUInteger("156765137136734", 8).result == "156765137136734")
        self.assertTrue(lexUInteger("f123ABCDEF13", 16).result == "f123ABCDEF13")
        
    def testRadix(self):
        self.assertTrue(lexRadix("#b", 2, LispAST_Number()))
        self.assertTrue(lexRadix("#o", 8, LispAST_Number()))
        self.assertTrue(lexRadix("#d", 10, LispAST_Number()))
        self.assertTrue(lexRadix("#x", 16, LispAST_Number()))
        self.assertFalse(lexRadix("#x", 2, LispAST_Number()))
        self.assertFalse(lexRadix("#o", 16, LispAST_Number()))

    def testExponentMarker(self):
        self.assertTrue(lexExponentMarker("e"))
        self.assertTrue(lexExponentMarker("s"))
        self.assertTrue(lexExponentMarker("f"))
        self.assertTrue(lexExponentMarker("d"))
        self.assertTrue(lexExponentMarker("l"))
        self.assertFalse(lexExponentMarker("b"))

    def testExactness(self):
        self.assertTrue(lexExactness("#i", LispAST_Number()))
        self.assertTrue(lexExactness("#e", LispAST_Number()))

    def testNumberPrefix(self):
        self.assertTrue(lexNumberPrefix("#x#e", 16, LispAST_Number()).result == "#x#e")
        self.assertTrue(lexNumberPrefix("#b#i", 2, LispAST_Number()))
        self.assertTrue(lexNumberPrefix("#i#o", 8, LispAST_Number()))
        self.assertTrue(lexNumberPrefix("#i#d", 10, LispAST_Number()))
        self.assertTrue(lexNumberPrefix("#i", 10, LispAST_Number()))
        self.assertTrue(lexNumberPrefix("", 10, LispAST_Number()))
        
    def testNumberSuffix(self):
        self.assertTrue(lexNumberSuffix("e+12345allo").result == "e+12345")
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
        self.assertTrue(lexURealNumber("123", 2).result != "123")

    def testRealNumber(self):
        self.assertTrue(lexRealNumber("+123.456e-10", 10).result == "+123.456e-10")
        self.assertTrue(lexRealNumber("-1234567", 8).result == "-1234567")
        self.assertTrue(lexRealNumber("+123456789abcdef", 16).result == "+123456789abcdef")
        self.assertTrue(lexRealNumber("+101010101111", 2).result == "+101010101111")
        self.assertTrue(lexRealNumber("+101210101111", 2).result != "+101210101111")

    def testNumberComplex(self):
        self.assertTrue(lexNumberComplex("123@456.789", 10, LispAST_Number()).result == "123@456.789")
        self.assertTrue(lexNumberComplex("-123.456+789.10i", 10, LispAST_Number()).result == "-123.456+789.10i")
        self.assertTrue(lexNumberComplex("-123.456-789.10e10i", 10, LispAST_Number()).result == "-123.456-789.10e10i")
        self.assertTrue(lexNumberComplex("-123.456e10+i", 10, LispAST_Number()).result == "-123.456e10+i")
        self.assertTrue(lexNumberComplex("-123.456e10-i", 10, LispAST_Number()).result == "-123.456e10-i")
        self.assertTrue(lexNumberComplex("+i", 10, LispAST_Number()).result == "+i")
        self.assertTrue(lexNumberComplex("+123.456e-10i", 10, LispAST_Number()).result == "+123.456e-10i")
        self.assertTrue(lexNumberComplex("-123.456e-10i", 10, LispAST_Number()).result == "-123.456e-10i")
        self.assertTrue(lexNumberComplex("10011-101100i", 2, LispAST_Number()).result == "10011-101100i")
        self.assertTrue(lexNumberComplex("1234567+1234567i", 8, LispAST_Number()).result == "1234567+1234567i")
        self.assertTrue(lexNumberComplex("-123456789abcdefi", 16, LispAST_Number()).result == "-123456789abcdefi")

    def testNumberBase(self):
        self.assertTrue(lexNumberBase("#d1234.5678e-9+543.21i", 10, LispAST_Number()).result == "#d1234.5678e-9+543.21i")
        self.assertTrue(lexNumberBase("#b10110", 2, LispAST_Number()).result == "#b10110")
        self.assertTrue(lexNumberBase("#o1234567-i", 8, LispAST_Number()).result == "#o1234567-i")
        self.assertTrue(lexNumberBase("#xabcdef-abc123i", 16, LispAST_Number()).result == "#xabcdef-abc123i")
        self.assertTrue(lexNumberBase("#e#d1234.5678e-9+543.21i", 10, LispAST_Number()).result == "#e#d1234.5678e-9+543.21i")

    def testNumber(self):
        self.assertTrue(lexNumber("#e1234.567e-89-i").result == "#e1234.567e-89-i");
        self.assertTrue(lexNumber("#i1234.567e-89-i").result == "#i1234.567e-89-i");
        self.assertTrue(lexNumber("-12.99e10i").result == "-12.99e10i");
        self.assertTrue(lexNumber("#b101011-011i").result == "#b101011-011i")
        self.assertTrue(lexNumber("#o#e1234/5676+77i").result == "#o#e1234/5676+77i")
        self.assertTrue(lexNumber("#xabcde/123-456abci").result == "#xabcde/123-456abci")
        self.assertTrue(lexNumber("#xabcde").result == "#xabcde")
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
        self.assertTrue(parseTokenType(parseTokens("1.23+5i"), LispTokenTypes.Number, LispAST_token))
        self.assertTrue(parseTokenType(parseTokens("(allo)"), LispTokenTypes.LParen, LispAST_token))
        self.assertTrue(parseTokenType(parseTokens(",@(allo)"), LispTokenTypes.CommaSplice, LispAST_token))

    def testParseSimpleDatum(self):
        self.assertTrue(parseSimpleDatum(parseTokens("allo")).rest == [])
        self.assertTrue(parseSimpleDatum(parseTokens("1.13e-13+i")).rest == [])
        self.assertTrue(parseSimpleDatum(parseTokens("#f")).rest == [])
        self.assertTrue(parseSimpleDatum(parseTokens("\"stringg \\\"str\\\" hello\"")).rest == [])

    def testParseList(self):
        self.assertTrue(parseList(parseTokens("(allo (sdf . asfsadf))")).rest == [])
        self.assertTrue(parseList(parseTokens("(car . cdr)")).rest == [])
        self.assertTrue(parseList(parseTokens("(car 1.13+i #\\a #f . (a b c 12 . ()))")).rest == [])

    def testParseVector(self):
        self.assertTrue(parseVector(parseTokens("#(1 2 3 4)")).rest == [])
        self.assertFalse(parseVector(parseTokens("#(1 2 3 . 4)")))
        self.assertFalse(parseVector(parseTokens("(1 2 3 4)")))

    def testParseCompoundDatum(self):
        self.assertTrue(parseCompoundDatum(parseTokens("'(allo #(1 2 3))")).rest == [])
        self.assertTrue(parseCompoundDatum(parseTokens("'(car . cdr)")).rest == [])
        self.assertTrue(parseCompoundDatum(parseTokens("`(one two)")).rest == [])
        self.assertTrue(parseCompoundDatum(parseTokens(",(one two)")).rest == [])
        self.assertTrue(parseCompoundDatum(parseTokens(",@(one two)")).rest == [])

    def testParseDatum(self):
        self.assertTrue(parseDatum(parseTokens(",@(allo ,(1 2 . nil) #(1 2 3))")).rest == [])
        self.assertFalse(parseDatum(parseTokens(",@(allo ,(1 2 . nil) #(1 2 . 3))")))

    def testParseQuotation(self):
        self.assertTrue(parseQuotation(parseTokens("'(allo 13 . 4)")).rest == [])
        self.assertTrue(parseQuotation(parseTokens("(quote (allo 13 4))")).rest == [])
        self.assertTrue(parseQuotation(parseTokens("(quote `(allo 13 4))")).rest == [])

    def testParseVariable(self):
        self.assertTrue(parseVariable(parseTokens("allo")))
        self.assertTrue(parseVariable(parseTokens("allo?")))
        self.assertTrue(parseVariable(parseTokens("<allo>?")))
        self.assertTrue(parseVariable(parseTokens("<define>?")))
        self.assertTrue(parseVariable(parseTokens("defin?")))
        self.assertFalse(parseVariable(parseTokens("define?")))
        self.assertFalse(parseVariable(parseTokens("if?")))

    def testParseSelfEvaluating(self):
        self.assertTrue(parseSelfEvaluating(parseTokens("#t")))
        self.assertTrue(parseSelfEvaluating(parseTokens("#f")))
        self.assertTrue(parseSelfEvaluating(parseTokens("1.134e-10-123i")))
        self.assertTrue(parseSelfEvaluating(parseTokens("#b1010011")))
        self.assertTrue(parseSelfEvaluating(parseTokens("#xABCDEF")))
        self.assertTrue(parseSelfEvaluating(parseTokens("#\\?")))
        self.assertTrue(parseSelfEvaluating(parseTokens("\"hello \\\"world\\\"\"")))

    def testParseLitteral(self):
        self.assertTrue(parseLitteral(parseTokens("'(hello world)")))
        self.assertTrue(parseLitteral(parseTokens("(quote (hello world))")))
        self.assertTrue(parseLitteral(parseTokens("(quote '(hello world))")))
        self.assertFalse(parseLitteral(parseTokens("(quote '(hello world)")))
        self.assertTrue(parseLitteral(parseTokens("#t")))
        self.assertTrue(parseLitteral(parseTokens("#o1234567")))
        self.assertFalse(parseLitteral(parseTokens("#o12345678")))
        self.assertFalse(parseLitteral(parseTokens("#a")))

    def testParseProcedureCall(self):
        self.assertTrue(parseProcedureCall(parseTokens("(fact! 10)")))
        self.assertTrue(parseProcedureCall(parseTokens("(thunk)")))
        self.assertTrue(parseProcedureCall(parseTokens("(thunk  )")))
        self.assertTrue(parseProcedureCall(parseTokens("(fib 1 2)")))
        self.assertTrue(parseProcedureCall(parseTokens("(f (g x))")))
        self.assertTrue(parseProcedureCall(parseTokens("(print '(1 2 3 4 5))")))
        self.assertTrue(parseProcedureCall(parseTokens("(map (lambda (x) (+ x 1)) '(1 2 3 4 5))")))
        self.assertFalse(parseProcedureCall(parseTokens("(lambda (lambda (x) (+ x 1)) '(1 2 3 4 5))")))
        self.assertFalse(parseProcedureCall(parseTokens("(if #t 1 2)")))
        self.assertFalse(parseProcedureCall(parseTokens("()")))
        
    def testParseLambdaExpression(self):
        self.assertTrue(parseLambdaExpression(parseTokens("(lambda (x) x)")))
        self.assertTrue(parseLambdaExpression(parseTokens("(lambda (x) (+ x 1))")))
        self.assertTrue(parseLambdaExpression(parseTokens("(lambda (x y z) (+ x y z))")))
        self.assertTrue(parseLambdaExpression(parseTokens("(lambda (x . rest) (foldl + x rest))")))
        self.assertTrue(parseLambdaExpression(parseTokens("(lambda () (print \"hello world\"))")))
        self.assertTrue(parseLambdaExpression(parseTokens("(lambda () (print 'test) (newline) (print 'test2))")))
        
    def testParseConditional(self):
        self.assertTrue(parseConditional(parseTokens("(if #t 'yes 'no)")))
        self.assertTrue(parseConditional(parseTokens("(if #t 'yes)")))
        self.assertTrue(parseConditional(parseTokens("(if (test) '#(1 2 3) (lambda (x) x))")))
        self.assertFalse(parseConditional(parseTokens("(iff #t 'true 'false)")))
        self.assertFalse(parseConditional(parseTokens("(if #t)")))

    def testParseAssignement(self):
        self.assertTrue(parseAssignement(parseTokens("(set! x 'allo)")))
        self.assertTrue(parseAssignement(parseTokens("(set! ^abc^ (+ 1 2))")))
        self.assertTrue(parseAssignement(parseTokens("(set! t? (if #t 'true 'false))")))
        self.assertFalse(parseAssignement(parseTokens("(set!! x 1)")))
        self.assertFalse(parseAssignement(parseTokens("(set! 'x 1)")))
        self.assertFalse(parseAssignement(parseTokens("(set! (x) 1)")))

    # def testParseDerivedExpression(self):
    #     self.assertTrue(parseDerivedExpression(parseTokens("(cond (#t 'true))")))
    #     self.assertTrue(parseDerivedExpression(parseTokens("(cond (abc 'true)((deg) 'true))")))
    #     self.assertTrue(parseDerivedExpression(parseTokens("(cond (abc 'true)((deg) 'true) (else #f))")))
    #     self.assertTrue(parseDerivedExpression(parseTokens("(cond (#t))")))
    #     self.assertTrue(parseDerivedExpression(parseTokens("(cond (#t => (lambda (x) x)))")))
    #     self.assertTrue(parseDerivedExpression(parseTokens("(case (fn-call) ((allo) 'allo))")))
    #     self.assertTrue(parseDerivedExpression(parseTokens("(case v ((allo) 'allo)) (else (+ 1 2)))")))
    #     self.assertTrue(parseDerivedExpression(parseTokens("(and one two three four)")))
    #     self.assertTrue(parseDerivedExpression(parseTokens("(and)")))
    #     self.assertTrue(parseDerivedExpression(parseTokens("(or one two three four)")))
    #     self.assertTrue(parseDerivedExpression(parseTokens("(or)")))
    #     self.assertTrue(parseDerivedExpression(parseTokens("(let ((x 1)) x)")))
    #     self.assertTrue(parseDerivedExpression(parseTokens("(let ((x 1)(y 2)) (+ x y))")))
    #     self.assertTrue(parseDerivedExpression(parseTokens("(let ((x 1)(y 2)) (+ x y))")))
    #     self.assertTrue(parseDerivedExpression(parseTokens("(let v ((x 1)(y 2)) (+ x y))")))
    #     self.assertTrue(parseDerivedExpression(parseTokens("(let* ((x 1)(y 2)) (+ x y))")))
    #     self.assertTrue(parseDerivedExpression(parseTokens("(letrec ((x 1)(y 2)) (+ x y))")))
    #     self.assertTrue(parseDerivedExpression(parseTokens("(begin a b c d)")))
    #     self.assertTrue(parseDerivedExpression(parseTokens("(do ((x 0 1) (y 'a)) (#t 'yes) (pp 'a))")))
    #     self.assertTrue(parseDerivedExpression(parseTokens("(delay (lambda (x) x))")))
    #     # todo: add quasiquote tests

    # def testParseExpression(self):
    #     self.assertTrue(parseExpression(parseTokens("'(allo)")))
    #     self.assertTrue(parseExpression(parseTokens("(allo)")))
    #     self.assertTrue(parseExpression(parseTokens("(+ 1 2)")))
    #     self.assertTrue(parseExpression(parseTokens("(if #t 'y 'n)")))
    #     self.assertTrue(parseExpression(parseTokens("(let ((x #t)) (if x 'y (no)))")))

# parseMacroUse(tokens):
# parseMacroBlock(tokens):
# parseDefinition(tokens):
# parseQuasiQuotation(tokens):

    def testEval(self):
        val = lispEval(parseExpression(parseTokens("12e3+5i")).result[0])
        self.assertTrue(val.real.numerator == 12000.0)
        self.assertTrue(val.real.denominator == 1.0)
        self.assertTrue(val.imag.numerator == 5.0)
        self.assertTrue(val.imag.denominator == 1.0)
        
        val2 = lispEval(parseExpression(parseTokens("#b101")).result[0])
        self.assertTrue(val2.real.numerator == 5)
        self.assertTrue(val2.imag.numerator == 0)

        valFalse = lispEval(parseExpression(parseTokens("#f")).result[0])
        self.assertTrue(valFalse == False)

        valTrue = lispEval(parseExpression(parseTokens("#t")).result[0])
        self.assertTrue(valTrue == True)

        valCharZ = lispEval(parseExpression(parseTokens("#\\Z")).result[0])
        self.assertTrue(valCharZ == "Z")

        valStr = lispEval(parseExpression(parseTokens("\"hello World\"")).result[0])
        self.assertTrue(valStr == "hello World")
        

def runLispTests():
    unittest.TestLoader().loadTestsFromTestCase(TestLispLex).run(unittest.TextTestRunner(sys.stdout,True, 1).run(unittest.TestLoader().loadTestsFromTestCase(TestLispLex)))
