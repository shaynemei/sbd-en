#!/usr/bin/env python3
# English tokeniser based on regex and an abbreviation list

import re
import sys

def main():
	# Read abbrev_list from file
	with open("abbrev_list") as file_abbrev:
	    data_abbrev = file_abbrev.read()
	    # transform the abbrev_list to a patter for regex
	    add_or = "|".join(data_abbrev.split("\n"))
	    abbrev = "".join([("\\" if token=="." else "")+token for token in add_or])

	# Read input text from cat command
	data = sys.stdin.read()

	# tokenise text with abbrev_list and regex
	tokenised = re.findall(abbrev+"|'ve|'s|'re|'m|n't|[a-zA-Z]+(?=n't)|\w+-\w+|\w+\/\w+|(?:[A-Z]\.)+|[A-Z][a-z][A-Z][a-z]+|[A-Z][a-z]+|[a-z]+|[A-Z]+|\d{2,4}s|\d{4}|\d+\.\d+|\d{1,3}(?:,?\d{3})*|--|[.,;:%&$#?(){}\"']|\n|[^ ]",data)

	# find quotation marks to further differentiate opening and closing (following gold)
	quotation_pos = [i for i, token in enumerate(tokenised) if token=="\""]
	for idx, pos in enumerate(quotation_pos):
	    if(idx%2==0):
	        tokenised[pos] = "``" # change opening quotations into double back-ticks
	    else:
	        tokenised[pos] = "''" # change closing quotations into two single straight quotations

	# join results while stripping the extra space leading new lines
	result = "".join([("" if tokenised[idx-1]=="\n" else " ")+token for idx, token in enumerate(tokenised)]).strip()
	result += " "
	print(result)

if __name__== "__main__":
  main()