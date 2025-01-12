import streamlit as st
import turbopuffer as tpuf
import os
from datetime import datetime, timedelta, date
from dotenv import load_dotenv
from typing import Dict, List, Optional, Any, Iterable
from pydantic import BaseModel
import instructor
import google.generativeai as genai


load_dotenv()

tpuf.api_key = os.getenv("TURBOPUFFER_API_KEY")
tpuf.api_base_url = "https://gcp-us-central1.turbopuffer.com"

# Set up the Streamlit App
st.title("AI Customer Support Agent with Memory 🛒")
st.caption("Chat with a customer support assistant who remembers your past interactions.")


class ShippingAddress(BaseModel):
    street: str
    city: str
    state: str
    zip: str

class Preferences(BaseModel):
    payment: str
    delivery: str
    communication: str

class CustomerProfile(BaseModel):
    customerId: str
    name: str
    email: str
    phone: str
    dob: str
    shippingAddress: ShippingAddress
    preferences: Preferences

class Product(BaseModel):
    productName: str
    price: float
    quantity: int

class Order(BaseModel):
    orderId: str
    orderDate: date
    deliveryDate: date
    products: List[Product]
    status: str

class CustomerServiceInteraction(BaseModel):
    interactionId: str
    orderId: str
    date: date
    description: str

class CustomerData(BaseModel):
    customerProfile: CustomerProfile
    orderHistory: List[Order]
    customerServiceInteractions: List[CustomerServiceInteraction]

class CustomerSupportAIAgent:
    def __init__(self):
        """Initialize the customer support agent with TurboPuffer."""
        tpuf.api_key = os.environ['TURBOPUFFER_API_KEY']
        
        # Create namespaces
        self.ns = tpuf.Namespace('customer-support-agent')
        
        self.client = instructor.from_gemini(
            client=genai.GenerativeModel(
                model_name="models/gemini-1.5-flash-latest",
            )
        )

    def handle_query(self, query: str, user_id: Optional[str] = None) -> str:
        """Handle a customer query by searching relevant memories and generating a response."""

        # Search for relevant context using hybrid search
        relevant_history = self.ns.query(
            top_k=5,
            filters=["And", [["user_id", "Eq", user_id]]],
            include_attributes=["query"]
        )
        
        print("relevant_history", relevant_history)
        
        # Build context from results
        context = "Relevant past information:\n"
        for memory in relevant_history:
            context += f"- {memory.attributes['message']}\n"
            
        print("context", context)

        # generate response using client
        response = self.client.chat.completions.create(
            model="gemini-1.5-flash-latest",
            messages=[
                {"role": "system", "content": "You are a helpful customer support agent."},
                {"role": "user", "content": query},
                {"role": "assistant", "content": context}
            ]
        )
        
        answer = response.choices[0].message.content
        
        print("answer", answer)
        
        
        # store the query and answer in the namespace
        self.ns.upsert(
            ids=[f"{user_id}-{datetime.now().timestamp()}"],
            vectors=[self.get_embeddings(query), self.get_embeddings(answer)],
            distance_metric="cosine_distance",
            attributes={"conversation": [query, answer],
                        "user_id": user_id, 
                        "role": ["user", "assistant"], 
                        "timestamp": datetime.now().timestamp(),
                        },
            schema={
                'content': {
                    'type': 'string',
                    'full_text_search': True
                }
            }
        )
        
        return answer
    
    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for a given text."""
        try:
            response = self.client.embeddings.create(
                model="models/embedding-001",
                input=text
            )
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return None
        
        return response.data[0].embedding

    def get_memories(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve all memories for a given user."""
        memories = self.chat_history.query(
            text="*",  # Match all
            top_k=100,
            filters=["And", [["user_id", "Eq", user_id]]],
        )
        
        return [memory.attributes for memory in memories]
    
    def generate_synthetic_data(self, user_id: Optional[str] = None):
        """Generate synthetic customer data for a given user."""
        today = datetime.now()
        order_date = (today - timedelta(days=10)).strftime("%B %d, %Y")
        expected_delivery = (today + timedelta(days=2)).strftime("%B %d, %Y")
        
        prompt = f"""Generate a detailed customer profile and order history for a TechGadgets.com customer with ID {user_id}.
            When generating the recent order, make it a high-end electronic device (placed on {order_date}, to be delivered by {expected_delivery})"""
            
        customer_data = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a data generation AI that creates realistic customer profiles and order hitories"},
                {"role": "user", "content": prompt}
            ],
            response_model=CustomerData
        )
                
        #customer_data = response.choices[0].message.parsed
        
        print("customer_data", customer_data)
        
        self.ns.upsert(
            ids=[f"{user_id}-{datetime.now().timestamp()}"],
            vectors=[self.get_embeddings(customer_data)],
            distance_metric="cosine_distance",
            attributes={"customer_data": [customer_data], 
                        "user_id": user_id, 
                        "role": "assistant", 
                        "timestamp": datetime.now().timestamp(),
                        },
            schema={
                'content': {
                    'type': 'string',
                    'full_text_search': True
                }
            }
        )
 

# Initialize the CustomerSupportAIAgent
support_agent = CustomerSupportAIAgent()

# Sidebar for customer ID and memory view
st.sidebar.title("Enter your Customer ID:")
previous_customer_id = st.session_state.get("previous_customer_id", None)
customer_id = st.sidebar.text_input("Enter your Customer ID")

if customer_id != previous_customer_id:
    st.session_state.messages = []
    st.session_state.previous_customer_id = customer_id
    st.session_state.customer_data = None

# Add button to generate synthetic data
if st.sidebar.button("Generate Synthetic Data"):
    if customer_id:
        with st.spinner("Generating customer data..."):
            st.session_state.customer_data = support_agent.generate_synthetic_data(customer_id)
        st.sidebar.success("Synthetic data generated successfully!")
    else:
        st.sidebar.error("Please enter a customer ID first.")

if st.sidebar.button("View Customer Profile"):
    if st.session_state.customer_data:
        st.sidebar.json(st.session_state.customer_data)
    else:
        st.sidebar.info("No customer data generated yet. Click 'Generate Synthetic Data' first.")

if st.sidebar.button("View Memory Info"):
    if customer_id:
        memories = support_agent.get_memories(user_id=customer_id)
        if memories:
            st.sidebar.write(f"Memory for customer **{customer_id}**:")
            if memories and "results" in memories:
                for memory in memories["results"]:
                    if "memory" in memory:
                        st.write(f"- {memory['memory']}")
        else:
            st.sidebar.info("No memory found for this customer ID.")
    else:
        st.sidebar.error("Please enter a customer ID to view memory info.")

# Initialize the chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
query = st.chat_input("How can I assist you today?")

if query and customer_id:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate and display response
    answer = support_agent.handle_query(query, user_id=customer_id)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

elif not customer_id:
    st.error("Please enter a customer ID to start the chat.")