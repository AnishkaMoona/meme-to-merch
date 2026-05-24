import streamlit as st
import pandas as pd
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from openai import OpenAI
import os
from datetime import datetime

st.set_page_config(page_title="Meme-to-Merch", page_icon="👕", layout="wide")
st.title("👕 Meme-to-Merch Trend Trader")
st.caption("Autonomous system that turns viral trends into merch products")

# Sidebar for API keys
with st.sidebar:
    openai_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    tavily_key = st.text_input("Tavily API Key (optional)", type="password")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

if not openai_key:
    st.warning("Please enter your OpenAI API key in the sidebar.")
    st.stop()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.85)

# Mock + Real Trend Fetch
def fetch_trends():
    mock_trends = [
        {"topic": "AI models hallucinating recipes", "score": 98, "subtext": "Putting glue on pizza is trending"},
        {"topic": "Corporate jargon bingo", "score": 94, "subtext": "Let's circle back on this"},
        {"topic": "Midnight laundry procrastination", "score": 89, "subtext": "3 AM cleaning instead of sleeping"},
    ]
    return pd.DataFrame(mock_trends)

# Agents
trend_scraper = Agent(
    role="Trend Scraper",
    goal="Identify currently viral, funny, or relatable topics suitable for merch.",
    backstory="You scan public internet culture and surface high-potential trends for streetwear.",
    llm=llm,
    verbose=True
)

satirist = Agent(
    role="Satire & Copywriting Expert",
    goal="Create extremely funny, wearable slogans and detailed DALL-E prompts.",
    backstory="Sharp-witted meme lord who has designed 100+ viral t-shirts.",
    llm=llm,
    verbose=True
)

designer = Agent(
    role="Visual Merch Designer",
    goal="Create highly detailed prompts optimized for DALL-E 3 apparel mockups.",
    backstory="Professional streetwear graphic designer focused on clean, bold prints.",
    llm=llm,
    verbose=True
)

pricer = Agent(
    role="Pricing & Market Strategist",
    goal="Recommend pricing tiers, target demographics, and go-to-market advice.",
    backstory="E-commerce analyst who understands margins, perceived value, and student/Gen-Z buyers.",
    llm=llm,
    verbose=True
)

# Tasks
def create_crew(topic):
    task1 = Task(
        description=f"Analyze this trend: {topic}. Generate 3 distinct merch concepts. For each concept output: slogan, target vibe, and a very detailed DALL-E 3 prompt for a t-shirt/hoodie graphic.",
        expected_output="JSON list of 3 concepts with keys: concept_name, slogan, dalle_prompt, vibe",
        agent=satirist
    )
    
    task2 = Task(
        description="For the generated concepts, recommend pricing tiers (Budget $19-25, Premium $29-39, Hype $45+), target buyer persona, and expected sell-out potential.",
        expected_output="Markdown summary with pricing and positioning advice",
        agent=pricer
    )
    
    crew = Crew(
        agents=[satirist, pricer],
        tasks=[task1, task2],
        process=Process.sequential,
        verbose=True
    )
    return crew

# UI
st.subheader("Current Viral Trends")
df = fetch_trends()
st.dataframe(df, use_container_width=True)

selected = st.selectbox("Pick a trend to transform into merch:", df['topic'].tolist())

if st.button("🚀 Run Full Agent Pipeline", type="primary"):
    with st.spinner("Agents are brainstorming slogans, visuals, and pricing..."):
        crew = create_crew(selected)
        result = crew.kickoff()
        
        st.success("Pipeline Complete!")
        st.markdown("### Agent Output")
        st.markdown(result)
        
        # Image Generation
        st.subheader("🎨 Generated Visual Mockups")
        client = OpenAI()
        
        sample_prompt = f"Create a bold, minimalist streetwear t-shirt graphic about {selected}. Clean vector style, isolated on transparent background, highly detailed, trending design."
        
        try:
            image_response = client.images.generate(
                model="dall-e-3",
                prompt=sample_prompt,
                size="1024x1024",
                quality="standard",
                n=1
            )
            st.image(image_response.data[0].url, caption="DALL-E 3 Merch Mockup", width=500)
        except Exception as e:
            st.error(f"Image generation failed: {e}")

st.info("Pro Tip: In a real version, connect Tavily or Reddit API in the Trend Scraper Agent for live data.")