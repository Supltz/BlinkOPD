import re
from math import *

from latex2sympy2 import latex2sympy


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def eval_tuple(s):
    sl = s[1:-1].split(',')

    try:
        if s[0] == '(' and s[-1] == ')' and len(sl) > 1:
            s = ','.join([str(round(eval(str(latex2sympy(sub))),2))
                          if 'infty' not in sub and sub not in ['a', '-a'] else sub for sub in sl])
            return f"({s})"

        elif s[0] == '[' and s[-1] == ']' and len(sl) > 1:
            s = ','.join([str(round(eval(str(latex2sympy(sub))),2))
                          if 'infty' not in sub and sub not in ['a', '-a'] else sub for sub in sl])
            return f"[{s}]"

    except Exception:
        return s

    return s


def is_equal(asw: str, gt_asw: str) -> bool:
    asw = asw.lower()
    gt_asw = gt_asw.lower()

    if asw.replace(' ', '') == '' or gt_asw.replace(' ', '') == '':
        return False

    if gt_asw.strip() == asw.strip():
        return True

    asw = eval_tuple(asw)
    gt_asw = eval_tuple(gt_asw)

    if gt_asw == asw:
        return True

    try:
        if round(eval(str(latex2sympy(gt_asw))), 2) == round(eval(str(latex2sympy(asw))), 2):
            return True

        else:
            return False
    except:
        return False


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]

    if len(substrs) > 1:
        substrs = substrs[1:]

        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string

                a = substr[0]
                b = substr[1]

                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b

    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string

    a, b = string.split("/")

    try:
        a = int(a)
        b = int(b)

        assert string == "{}/{}".format(a, b)

        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string

    except:
        return string


def _remove_right_units(string):
    splits = string.split("\\text{ ")

    return splits[0]


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string

    splits = string.split("\\sqrt")

    new_string = splits[0]

    for split in splits[1:]:
        if len(split) > 0 and split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr

    return new_string


def _strip_string(string):
    string = string.replace("\n", "")

    string = string.replace("\\!", "")

    string = string.replace("\\\\", "\\")

    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    string = string.replace("\\$", "")
    string = string.replace("$", "")

    string = _remove_right_units(string)

    string = string.replace("\\%", "")
    string = string.replace("\\%", "")

    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    if len(string.split("=")) == 2:
        string = string.split("=")[-1]
    if len(string.split("\\approx")) == 2:
        string = string.split("\\approx")[-1]

    if 'sqrt' in string:
        string = _fix_sqrt(string)

    string = string.replace(" ", "")

    if 'sqrt' in string:
        string = _fix_fracs(string)

    if string == "0.5":
        string = "\\frac{1}{2}"

    string = _fix_a_slash_b(string)

    return string


def find_math_answer(s: str) -> str:
    s = s.lower()
    if '{}' in s:
        s = s.replace('{}', '')

    try:
        pattern = re.compile('oxed{(.*)}', flags=re.S)
        ans = pattern.findall(s)[-1]
    except:
        ans = s

    if ans.find('}') != -1 and (ans.find('{') == -1 or  ans.find('}') < ans.find('{')):
        ans = ans.split('}')[0]

    ans = ans.split('=')[-1]
    ans = ans.split('\\approx')[-1]

    ans = ans.replace(" ", "").replace("\\,", "").replace('∞', '\\infty')
    ans = ans.replace("+\\infty", "\\infty").replace("\\\\", "\\").replace("\n", "")
    ans = ans.replace('\\text', '').replace('\\mbox', '').replace('bmatrix', 'pmatrix')
    ans = ans.replace("\\left", "").replace('\\right', '').replace("^{\\circ}", "")
    ans = ans.replace("^\\circ", "").replace("{m}^3", "").replace("m^3", "")
    ans = ans.replace("{units}", "").replace("units", "").replace("{km}", "").replace("km", "")

    return _strip_string(ans)
