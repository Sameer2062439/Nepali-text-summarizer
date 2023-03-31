import string


class InputValidator:
    def __init__(self, file_content):
        self.file_content = file_content
        self.text = ''
        self.eng_text = ''

    def detect_language(self,character):
        maxchar = max(character)
        if (u'\u0900' <= maxchar <= u'\u097f') or (maxchar == u"\u0020"):
            return True

    def validate_to_var(self):
        # print(self.file_content)
        for index, i in enumerate(self.file_content):
            # for i in j:
            # print(i)
            isNep = self.detect_language(i)
            if isNep == True:
                # print(i,end="\t")
                # print(isEng)
                self.text = self.text + i
            else:
                self.eng_text = self.eng_text + i
            # print(self.text)
                # print(f"not nepali {index} : {self.eng_text.encode()}")
        # print(f"not nepali : {self.eng_text}")
        return self.text
    
    def validate_to_file(self):
        for i in self.file_content:
            isEng = self.detect_language(i)
            if isEng == True:
                with open('out.txt','a', encoding='utf-8') as f:
                    f.write(i)