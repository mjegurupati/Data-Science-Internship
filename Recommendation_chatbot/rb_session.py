from rb_actions import input_processor, intentIdentifier, check_required_params, check_actions
from rb_context import FirstGreeting
from rb_intents import IntentComplete
import pandas as pd


class Session:
    '''Initialise a default session'''
    def __init__(self, attributes=None, active_contexts=[FirstGreeting(), IntentComplete() ]):

        # Active contexts not used yet, can use it to have multiple contexts
        self.active_contexts = active_contexts

        # Contexts are flags which control dialogue flow
        self.context = FirstGreeting()

        # Intent tracks the current state of dialogue
        #self.current_intent = First_Greeting()
        self.current_intent = None

        # attributes hold the information collected over the conversation
        self.attributes = {}

    '''Not used yet, but is intended to maintain active contexts'''
    def update_contexts(self):

        for context in self.active_contexts:
            if context.active:
                context.decrease_lifespan()

    '''Generate response to user input'''
    def reply(self, user_input):

        self.attributes, clean_input = input_processor(user_input, self.context, self.attributes, self.current_intent)
        #print(clean_input)                
        
                                                

        self.current_intent = intentIdentifier(clean_input, self.context, self.current_intent)

        prompt, self.context = check_required_params(self.current_intent, self.attributes, self.context)
        

        # prompt is None means all parameters satisfied, perform the intent action
        if prompt is None:
            if self.context.name != 'IntentComplete':
                prompt, self.context = check_actions(self.current_intent, self.attributes, self.context)
                #print("reply - prompt ", prompt, " context ", self.context,"attributes",self.attributes)
                #print(self.attributes.keys())
                #L=self.attributes
                #print(L)
                #type(L)
                '''l=[]
                for entities in self.attributes:
                        l.append(self.attributes[entities])
               
                print(l)'''
                
                
                    
                '''TODO : YOUR CODE HERE TO GET RECOMMENDATION BASED ON THE ENTITY VALUES'''
                #df=pd.DataFrame(data=d,columns= ['DISEASES', 'OPERATION','CONSULTATION_FEE','LOCATION'])
                
                
                if prompt == "Hospital_Reommendation" :
                    #l=[]
                    
                    #l.append(self.attributes.values())
                    df1=pd.read_csv('C:/Users/Pakhala/PycharmProjects/AIMLchatbot/bot/Hospitals.csv')
                    #print(l[0])
                    for row in df1.itertuples():
                        if(row.DISEASES==self.attributes['DISEASES']):
                           if(str(row.OPERATION)==self.attributes['OPERATION']):
                                
                                if(str(row.CONSULTATION_FEE)==self.attributes['CONSULTATION_FEE']):
                                    if(row.LOCATION==self.attributes['LOCATION']):
                                        x=str(row.RECOMMENDATION)
                                        print("we recommend you to join in  ",x)
                                   
                    
                                        
                                       
                                        
                    
                elif prompt == "Game_Recommendation" :
                    df1=pd.read_csv('C:/Users/Pakhala/PycharmProjects/AIMLchatbot/bot/game_recommendation.csv')
                    #print(l[0])
                    for row in df1.itertuples():
                        if(row.Genre==self.attributes['Genre1']):
                           if(str(row.Age)==self.attributes['Age']):

                                if(str(row.Platform)==self.attributes['Platform']):

                                    if(str(row.Gender)==str(self.attributes['Gender'])):
                                        print("we suggest you to play ",row.Recommendation)
                                        break

               
                 
                        
                    #print("Your book is issued and ready to collect")
                 
                                    
                prompt = "BOT: Hi! How may I assist you?"

        # Resets the state after the Intent is complete
        if self.context.name == 'IntentComplete':
            self.attributes = {}
            self.context = FirstGreeting()
            self.current_intent = None

        return prompt
