import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch import nn

tokenizer = get_tokenizer('basic_english')
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

dataset = [
    ("Recept", "Chocolate Chip Cookies, Preheat the oven to 350°F (175°C), In a mixing bowl, cream together 1 cup of softened butter, 1 cup of white sugar, and 1 cup of packed brown sugar, Beat in 2 large eggs, one at a time, then stir in 2 teaspoons of vanilla extract, Dissolve 1 teaspoon of baking soda in 2 teaspoons of hot water, Add to the batter along with ½ teaspoon of salt, Gradually mix in 3 cups of all-purpose flour, Stir in 2 cups of semisweet chocolate chips, Drop dough by rounded spoonfuls onto ungreased cookie sheets, Bake for 10 to 12 minutes until edges are nicely browned, Allow cookies to cool on baking sheet for 5 minutes before transferring to a wire rack to cool completely."),
    ("Recept", "Classic Macaron, Preheat the oven to 300°F (150°C) and line baking sheets with parchment paper, In a food processor, pulse 1 cup of almond flour and 1 3/4 cups of powdered sugar until combined, Sift the mixture into a large bowl, In a separate bowl, beat 3 large egg whites until foamy, Gradually add 1/4 cup of granulated sugar, beating until stiff peaks form, Gently fold the egg white mixture into the almond flour mixture until smooth and shiny, Transfer the batter to a piping bag fitted with a round tip, Pipe small circles onto the prepared baking sheets, Let the macarons sit at room temperature for 30 minutes to form a skin, Bake for 15-18 minutes, until macarons are set but not browned, Let cool completely on the baking sheets, For the filling, sandwich pairs of macarons together with your favorite filling, such as ganache, buttercream, or jam."),
    ("Recept", "Spaghetti Carbonara, Cook 8 ounces of spaghetti according to package instructions until al dente, While pasta is cooking, in a large skillet, cook 4 ounces of diced pancetta over medium heat until crispy, Remove pancetta from skillet and set aside, In the same skillet, add 2 minced garlic cloves and cook until fragrant, about 1 minute, In a bowl, whisk together 2 large eggs and 1/2 cup of grated Parmesan cheese, Drain cooked spaghetti and add it to the skillet with garlic, Off heat, quickly toss spaghetti with garlic, then pour in egg and cheese mixture, stirring quickly to coat spaghetti evenly, Add cooked pancetta back to skillet and toss to combine, Season with salt and black pepper to taste, Serve immediately with additional grated Parmesan cheese on top if desired."),
    ("Recept", "Grilled Chicken Salad, Preheat grill to medium-high heat, Season boneless, skinless chicken breasts with salt, black pepper, and olive oil, Grill chicken for 6-8 minutes per side until cooked through, Set aside to cool, In a large bowl, combine mixed greens, cherry tomatoes, sliced cucumber, sliced red onion, and any other desired salad vegetables, Slice grilled chicken into strips and add to the salad, In a small bowl, whisk together olive oil, balsamic vinegar, Dijon mustard, salt, and black pepper to make the dressing, Pour dressing over salad and toss to coat evenly, Serve immediately as a delicious and nutritious meal."),
    ("Recept", "Homemade Pizza, Preheat oven to 475°F (245°C), Roll out pizza dough on a lightly floured surface to desired thickness, Transfer dough to a pizza stone or baking sheet lined with parchment paper, Spread pizza sauce evenly over dough, leaving a small border around the edges, Sprinkle shredded mozzarella cheese over sauce, Add desired toppings such as pepperoni, sliced bell peppers, mushrooms, and olives, Bake pizza in preheated oven for 12-15 minutes until crust is golden brown and cheese is bubbly, Remove from oven and let cool for a few minutes before slicing, Serve hot and enjoy your homemade pizza!"),
    ("Recept", "Chicken Alfredo Pasta, Cook 8 ounces of fettuccine according to package instructions until al dente, While pasta is cooking, season boneless, skinless chicken breasts with salt and black pepper, In a large skillet, heat olive oil over medium-high heat and cook chicken until golden brown and cooked through, about 6-8 minutes per side, Remove chicken from skillet and let rest, In the same skillet, melt butter over medium heat and add minced garlic, Cook for 1-2 minutes until fragrant, Stir in heavy cream and bring to a simmer, Add grated Parmesan cheese and stir until melted and smooth, Season sauce with salt, black pepper, and a pinch of nutmeg, Slice cooked chicken and add it back to the skillet along with cooked fettuccine, Toss everything together until pasta is coated in sauce, Serve hot with additional grated Parmesan cheese and chopped parsley on top."),
    ("Recept", "STEP 1 Bring a pan of water to the boil and carefully lower in the eggs. Cook for 6 mins, then cool under running water until they can be peeled. Peel the eggs, then leave to cool completely. STEP 2 Mash or chop the eggs, then mix with 1½ tbsp mayonnaise and some seasoning, if you like. Toast the bread. STEP 3 Lay one slice of bread on a board. Butter it, then spread on three quarters of the egg and scatter over the cress. Add another slice of toast and gently spread on the remaining mayo. Add the tomato or lettuce and ham or cheese (or whichever combination you prefer). Dot the remaining egg over the top, spread gently, then top with the final piece of toast. Cut the crusts off if you like, then gently cut the sandwich into four quarters, being careful not to squash out the egg. Skewer each sandwich with a sandwich pick. Serve with crisps."),
    ("Recept", "STEP 1 Preheat the oven to 350°F. STEP 2 In a large mixing bowl, combine flour, sugar, and baking powder. In a separate bowl, whisk together eggs and milk. Pour the wet ingredients into the dry ingredients and mix until smooth. STEP 3 Grease a baking pan and pour the batter into it. Bake in the preheated oven for 25-30 minutes, or until a toothpick comes out clean. STEP 4 Allow the cake to cool, then frost and decorate as desired."),
    ("Recept", "STEP 1 Start by marinating the chicken with your choice of spices and let it sit for at least 30 minutes. STEP 2 Heat oil in a pan and add the marinated chicken. Cook until it's browned and cooked through. STEP 3 In a separate pot, cook the rice. STEP 4 Assemble the dish by layering the cooked rice, chicken, and your favorite toppings. Serve hot."),
    ("Reminder", "Buy groceries: eggs, milk, bread, and bananas"),
    ("Reminder", "Call mom to wish her a happy birthday"),
    ("Reminder", "Schedule dentist appointment for next week"),
    ("Reminder", "Pay rent before the end of the month"),
    ("Reminder", "Attend meeting with project team at 10:00 AM tomorrow"),
    ("Reminder", "Renew car insurance policy by the end of the month"),
    ("Reminder", "Pick up dry cleaning on Thursday afternoon"),
    ("Reminder", "Submit report to manager by Friday EOD"),
    ("Reminder", "Attend yoga class at 6:00 PM today"),
    ("Reminder", "Take out the trash before leaving for work"),
    ("Reminder", "Send birthday gift to friend via mail"),
    ("Reminder", "Water the plants in the garden"),
    ("Reminder", "Check and reply to important emails"),
    ("Reminder", "Review notes for upcoming exam"),
    ("Reminder", "Visit the library to return borrowed books"),
    ("Reminder", "Call the plumber to fix the leaky faucet"),
    ("Reminder", "Set aside time for exercise every day"),
    ("Reminder", "Update resume and LinkedIn profile"),
    ("Reminder", "Start reading the new book purchased last week"),
    ("Reminder", "Plan weekend getaway with family"),
    ("Reminder", "23:23 - go to sleep"),
    ("Link", "https://www.google.com/"),
    ("Link", "https://www.youtube.com/"),
    ("Link", "https://www.wikipedia.org/"),
    ("Link", "Check out this article on the https://www.nasa.gov/ website."),
    ("Link", "Find the latest news on https://www.cnn.com/."),
    ("Link", "Shop for a wide variety of products on https://www.amazon.com/."),
    ("Link", "Connect with friends and family on https://www.facebook.com/."),
    ("Link", "Follow me on https://twitter.com/?lang=en for updates."),
    ("Link", "Share your photos on https://www.instagram.com/."),
    ("Link", "Watch videos on https://www.youtube.com/."),
    ("Link", "Listen to music on https://open.spotify.com/."),
    ("Link", "Read books on https://www.amazon.com/."),
    ("Link", "Learn a new skill on https://www.coursera.org/."),
    ("Link", "Find a job on https://www.linkedin.com/login."),
    ("Personal Info", "Name: Emily Johnson"),
    ("Personal Info", "Date of Birth: January 20, 1995"),
    ("Personal Info", "Address: 456 Oak Avenue, Smalltown, Canada"),
    ("Personal Info", "Phone Number: +1 (555) 987-6543"),
    ("Personal Info", "Email: emily.johnson@example.com"),
    ("Personal Info", "Occupation: Nurse"),
    ("Personal Info", "Blood Type: A-"),
    ("Personal Info", "Mother's Maiden Name: Brown"),
    ("Personal Info", "Social Security Number: 987-65-4321"),
    ("Personal Info", "Favorite Color: Green"),
    ("Personal Info", "Favorite Food: Sushi"),
    ("Personal Info", "Favorite Movie: Inception"),
    ("Personal Info", "Pet: Cat named Luna"),
    ("Personal Info", "+37121231231"),
    ("Personal Info", "+8-800-555-35-35"),
    ("Personal Info", "example@google.com"),
    ("Personal Info", "example2313252@gmail.com"),
    ("Personal Info", "ada5dag2@inbox.lv"),
    ("Personal Info", "kautkas@example.ru")
    ]

vocab = build_vocab_from_iterator(yield_tokens(dataset), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: ["Recept", "Reminder", "Link", "Personal Info"].index(x)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
         label_list.append(label_pipeline(_label))
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

