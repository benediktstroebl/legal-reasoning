TAX_PROBLEMS_SYSTEM_PROMPT = """Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: [begin_of_thought] {thought with steps separated with '\\n\\n'} [end_of_thought] Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: [begin_of_solution] {final formatted, precise, and clear solution} [end_of_solution] Now, try to solve the following question through the above guidelines:"""
TAX_PROBLEMS_USER_PROMPT = """Please answer the following question. Your final solution should be concise and clear.\n\nQuestion: {question}"""
TAX_PROBLEMS_MULTIPLE_CHOICE_USER_PROMPT = """Please answer the following multiple-choice question. Your final solution should only contain the correct answer and not the index of the answer or any other information.\n\n
Question: {question}\n\n
Possible Answers:\n{answers}\n\n
"""


BAREXAM_SYSTEM_PROMPT = """Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\\n\\n'} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines:"""
BAREXAM_MULTIPLE_CHOICE_SYSTEM_PROMPT = """You are given a multiple-choice U.S. bar exam question. Your final answer should be the letter corresponding to the correct answer (for example B).

{prompt}{question}

Possible Answers:\n{answers}"""



convert_prompt = "Another solution is written in an unstructured way. Your job is to convert them into two sections: \
    <|begin_of_thought|> \
    (Thought process, you should copy exactly the thinking process of the original solution.) \
    <|end_of_thought|> \
    <|begin_of_solution|> \
    (Final letter of the correct answer. Hence, 1. -> you return 'A', 2. -> you return 'B', 3. -> you return 'C', 4. -> you return 'D' as the final solution, and so on. No additional information is needed.) \
    <|end_of_solution|> \
    Here is an example demonstration of a different question, you can refer to its format: \
    {example} \
    Important: You should almost copy all the contents word-by-word of the original solution. Just convert them into two sections. \
    Make sure you include: <|begin_of_slow_thought|>, <|end_of_slow_thought|>,  <|begin_of_solution|>,<|end_of_solution|>  These four headers explicitly.  \
    Content to be converted: {content}"  # noqa: E501

convert_prompt_example_original = (
    "<|begin_of_thought|>\n\n"
    "Okay, so I've got this problem here. Mr. Wang leaves home at 6 AM, riding his bike at 12 km/h, "
    "and he stops to rest for 6 minutes after every 30 minutes of riding. Then, when he arrives at a park "
    "that's 16.8 km away, I need to find out the angle between the hour and minute hands on his watch.\n\n"
    "Alright, first things first, I need to figure out how long it takes Mr. Wang to ride 16.8 km, including "
    "his rest periods.\n\n"
    "So, his speed is 12 km/h. To find out how long it takes to go 16.8 km without any stops, I can use the formula "
    "time = distance/speed. That would be 16.8 divided by 12, which is 1.4 hours. To make it easier, that's 1 hour and 24 minutes.\n\n"
    "But wait, he doesn't ride straight through. He stops for 6 minutes after every 30 minutes of riding. So, I need to see how many "
    "of those 30-minute riding periods are there in his total riding time.\n\n"
    "In 1 hour and 24 minutes of riding, how many 30-minute segments are there? Well, 1 hour is 60 minutes, plus 24 minutes makes 84 minutes "
    "total riding time. So, 84 divided by 30 is 2.8. That means he has two full 30-minute riding periods and a partial one.\n\n"
    "After each full 30-minute riding period, he rests for 6 minutes. So, for two full periods, he rests twice, which is 12 minutes of rest.\n\n"
    "Now, for the partial riding period. Since 2 times 30 minutes is 60 minutes, and he has 84 minutes of riding, the remaining riding time is 84 minus 60, "
    "which is 24 minutes. So, he rides for 24 minutes without another rest because he doesn't complete another 30-minute segment.\n\n"
    "So, total time taken is riding time plus rest time. That's 84 minutes riding plus 12 minutes resting, totaling 96 minutes.\n\n"
    "Wait a minute, but he stops after every 30 minutes of riding, but in the last partial period of 24 minutes, does he rest again? I think he only rests after "
    "completing 30 minutes of riding, so in this case, since the last riding period is only 24 minutes, he doesn't take an additional rest after that.\n\n"
    "So, total time should be 84 minutes riding plus 12 minutes resting, which is indeed 96 minutes, or 1 hour and 36 minutes.\n\n"
    "So, he leaves at 6 AM and takes 1 hour and 36 minutes to reach the park, arriving at 7:36 AM.\n\n"
    "Now, I need to find the angle between the hour and minute hands at 7:36.\n\n"
    "To find the angle between the hour and minute hands, I can use the formula:\n\n"
    "|30H - 5.5M|\n\n"
    "where H is the hour and M is the minutes.\n\n"
    "At 7:36, H is 7 and M is 36.\n\n"
    "So, plugging in:\n\n"
    "30*7 = 210\n\n"
    "5.5*36 = 198\n\n"
    "210 - 198 = 12\n\n"
    "So, the angle is 12 degrees.\n\n"
    "Wait, but I should make sure that's the smaller angle. Sometimes, the larger angle is considered, but usually, the smaller one is what is asked for.\n\n"
    "So, the angle between the hour and minute hands at 7:36 AM is 12 degrees.\n\n"
    "I think that's the answer.<|end_of_thought|>\n\n"
    "<|begin_of_solution|>\n\n"
    "Mr. Wang leaves home at 6 AM and rides at a speed of 12 km/h, stopping to rest for 6 minutes after every 30 minutes of riding. "
    "He arrives at a park 16.8 km away. To determine the angle between the hour and minute hands on his watch when he arrives, we first calculate the total time taken.\n\n"  # noqa: E501
    "1. **Riding time without stops**:\n\n"
    "$$\\text{Time} = \\frac{\\text{Distance}}{\\text{Speed}} = \\frac{16.8 \\text{ km}}{12 \\text{ km/h}} = 1.4 \\text{ hours} = 84 \\text{ minutes}$$\n\n"
    "2. **Rest periods**:\n\n"
    "  - He rests for 6 minutes after every 30 minutes of riding.\n\n"
    "  - In 84 minutes of riding, he completes 2 full 30-minute segments and a partial 24-minute segment.\n\n"
    "  - He rests twice, totaling 12 minutes of rest.\n\n"
    "3. **Total time**:\n\n"
    "$$\\text{Total time} = 84 \\text{ minutes (riding)} + 12 \\text{ minutes (rest)} = 96 \\text{ minutes} = 1 \\text{ hour and } 36 \\text{ minutes}$$\n\n"
    "  - He arrives at 7:36 AM.\n\n"
    "4. **Angle between hour and minute hands at 7:36**:\n\n"
    "  - Use the formula:\n\n"
    "$$\\text{Angle} = |30H - 5.5M|$$\n\n"
    "  - At 7:36, $H = 7$ and $M = 36$:\n\n"
    "$$\\text{Angle} = |30 \\times 7 - 5.5 \\times 36| = |210 - 198| = 12 \\text{ degrees}$$\n\n"
    "Thus, the angle between the hour and minute hands on his watch is $\\boxed{12}$.<|end_of_solution|>\n"  # noqa: E501
)

# From https://arxiv.org/pdf/2412.09413
system_prompt = "Your role as an assistant involves thoroughly exploring questions through a systematic long \
thinking process before providing the final precise and accurate solutions. This requires \
engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, \
backtracing, and iteration to develop well-considered thinking process. \
Please structure your response into two main sections: Thought and Solution. \
In the Thought section, detail your reasoning process using the specified format: \
<|begin_of_thought|> {thought with steps separated with '\n\n'} \
<|end_of_thought|> \
Each step should include detailed considerations such as analisying questions, summarizing \
relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining \
any errors, and revisiting previous steps. \
In the Solution section, based on various attempts, explorations, and reflections from the Thought \
section, systematically present the final solution that you deem correct. The solution should \
remain a logical, accurate, concise expression style and detail necessary step needed to reach the \
conclusion, formatted as follows: \
<|begin_of_solution|> \
{final formatted, precise, and clear solution} \
<|end_of_solution|> \
Now, try to solve the following question through the above guidelines:"


convert_prompt_example = """<|begin_of_thought|> \n\nOkay, so I've got this bar exam question here, and I need to figure it out step by step. Let's start by breaking down the scenario.\n\nFirst, Paul is the plaintiff in a personal injury case. He called Wes as a witness to testify that Dan's car ran a red light while Paul was riding in it. But Wes testified that Dan's car didn't run the light. So, Wes's testimony is conflicting with Paul's case. That means Wes's credibility might be an issue, right? Because if Wes is not reliable, Paul's case could be weakened.\n\nNow, Dan, the defendant, is calling Zemo as a witness. Dan asks Zemo if he knows Vic's reputation for veracity in the community where Vic resides. Wait, Vic hasn't been mentioned before. So Vic must be another witness, probably one that Paul called earlier. Maybe Vic testified about something related to the case, like the accident or Dan's actions.\n\nThe question is about whether the trial judge should rule this question as objectionable or not. The options are about whether it's collateral, about character evidence, impeachment, or whether Zemo knows Vic personally.\n\nLet me recall the rules of evidence, specifically about impeachment and reputation evidence. Under the Federal Rules of Evidence, a witness's credibility can be attacked by any evidence that would be admissible for that purpose, including evidence of the witness's character or character trait, such as truthfulness. However, character evidence is generally not admissible to prove that a person acted in accordance with that character on a particular occasion, except in certain cases.\n\nImpeachment allows a party to attack the credibility of a witness. One way to do this is by introducing evidence of the witness's reputation for truthfulness. So, if Vic is a witness who testified against Dan, Dan can impeach Vic by asking about his reputation for veracity.\n\nBut wait, the question is whether the question is objectionable. Let's look at the options.\n\nOption 1 says it's objectionable because it's collateral. Collateral matters are those that are not directly at issue in the case and cannot be proved by extrinsic evidence. However, reputation evidence for impeachment is not considered collateral because it goes to the witness's credibility, which is directly relevant.\n\nOption 2 says it's objectionable because character cannot be proven by generalities. But reputation evidence is about the community's view, which is a generality, but it's allowed for impeachment purposes. So this might not be the right objection.\n\nOption 3 says it's unobjectionable because it's a foundation for impeachment. That makes sense because if Vic is a witness, Zemo can testify about Vic's reputation for truthfulness to impeach him.\n\nOption 4 says it's unobjectionable because Zemo could be expected to know Vic personally. But the question is about reputation, not personal knowledge. So even if Zemo doesn't know Vic personally, if he knows about Vic's reputation in the community, that's sufficient.\n\nWait, but under FRE 608(a), a witness's character for truthfulness or untruthfulness can be attacked by evidence of the reputation of the witness in the community for having that character. So Zemo can testify about Vic's reputation, even if he doesn't know Vic personally, as long as he knows about the reputation.\n\nSo the question is whether the trial judge should allow this. The options are about whether the question is objectionable or not.\n\nOption 3 says it's unobjectionable because it's a foundation for impeachment. That seems correct because impeachment allows for reputation evidence.\n\nOption 4 is about whether Zemo knows Vic personally, but the question is about reputation, so it's not necessary for Zemo to know Vic personally. So option 4 is not the reason.\n\nTherefore, the correct ruling is that the question is unobjectionable because it's laying the foundation for impeachment of Vic. \n\n<|end_of_thought|>\n\n<|begin_of_solution|>C<|end_of_solution|>"""