"""
LangChain Agent Service

This module implements the agentic system using LangChain with custom tools.
The agent can:
- Search for relevant video segments
- Analyze pose data
- Validate against biomechanics rules
- Generate evidence-grounded answers
"""

import logging
from typing import List, Dict, Optional
import json
from pathlib import Path

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

from app.services.vector_db import vector_db
from app.services.pose_estimation import PoseEstimator, PoseData
from app.services.motion_analyzer import MotionAnalyzer
from app.core.config import settings

logger = logging.getLogger(__name__)


class MotionAnalysisAgent:
    """
    Agentic system for analyzing human motion in videos.
    
    This agent uses multiple tools to:
    1. Find relevant video segments (retrieval)
    2. Extract pose data (perception)
    3. Analyze biomechanics (reasoning)
    4. Generate grounded answers
    """
    
    def __init__(self):
        """Initialize the agent with tools and LLM"""
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.3  # Lower temperature for more factual responses
        )
        
        # Initialize helper services
        self.pose_estimator = PoseEstimator()
        self.motion_analyzer = MotionAnalyzer()
        
        # Create tools
        self.tools = self._create_tools()
        
        # Create agent
        self.agent = self._create_agent()
        
        # Create executor
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True
        )
    
    def _create_tools(self) -> List[Tool]:
        """
        Create the tools the agent can use.
        
        These tools implement the separation between perception and reasoning:
        - search_video_segments: Retrieval tool
        - get_pose_analysis: Perception tool
        - validate_safety: Reasoning tool
        - get_biomechanics_standards: Knowledge retrieval tool
        """
        tools = [
            Tool(
                name="search_video_segments",
                func=self._search_video_segments_tool,
                description="""
                Search for relevant video segments based on a query.
                Input should be a search query describing what to look for (e.g., "squat bottom position").
                Returns list of relevant video segments with timestamps and descriptions.
                Use this tool FIRST to find where relevant motion occurs in the video.
                """
            ),
            Tool(
                name="get_pose_analysis",
                func=self._get_pose_analysis_tool,
                description="""
                Get detailed pose analysis for a specific video segment.
                Input should be a segment ID from search_video_segments.
                Returns detailed joint angles, positions, and biomechanics measurements.
                Use this tool AFTER finding relevant segments to get quantitative data.
                """
            ),
            Tool(
                name="validate_safety",
                func=self._validate_safety_tool,
                description="""
                Validate motion safety against biomechanics standards.
                Input should be: "exercise_type|segment_id" (e.g., "squat|video1_001").
                Returns safety violations, risk levels, and specific measurements.
                Use this tool to check if motion meets safety guidelines.
                """
            ),
            Tool(
                name="get_biomechanics_standards",
                func=self._get_standards_tool,
                description="""
                Get biomechanics safety standards for an exercise type.
                Input should be exercise name (e.g., "squat", "pushup", "deadlift").
                Returns detailed safety standards with angle ranges and sources.
                Use this tool to understand what "correct form" means for an exercise.
                """
            )
        ]
        
        return tools
    
    def _create_agent(self):
        """Create the ReAct agent with custom prompt"""
        
        tools_str = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        tool_names_str = ", ".join([tool.name for tool in self.tools])

        template = f"""You are an expert biomechanics analyst helping coaches analyze human motion in videos.

You have access to these tools:
{tools_str}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names_str}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

CRITICAL INSTRUCTIONS:
1. ALWAYS use tools to gather evidence before answering
2. ALWAYS cite specific timestamps and measurements in your answer
3. ALWAYS check safety using the validate_safety tool
4. Use search_video_segments FIRST to find relevant moments
5. Then use get_pose_analysis to get measurements
6. Compare measurements against standards using validate_safety
7. Reference specific angles, positions, and timestamps in your final answer

When answering:
- Start with search_video_segments to find relevant clips
- Get detailed measurements with get_pose_analysis
- Validate safety with validate_safety
- Cite ALL evidence (timestamps, angles, measurements)
- Be specific and quantitative

Question: {{input}}

Thought: {{agent_scratchpad}}
"""
        
        prompt = PromptTemplate.from_template(template)
        # create_react_agent internally checks for these exact names in input_variables
        if "tools" not in prompt.input_variables:
            prompt.input_variables.append("tools")
        if "tool_names" not in prompt.input_variables:
            prompt.input_variables.append("tool_names")
        
        return create_react_agent(self.llm, self.tools, prompt)
    
    # Tool Implementation Methods
    
    def _search_video_segments_tool(self, query: str) -> str:
        """
        Tool: Search for relevant video segments.
        
        This implements the retrieval phase - finding relevant moments
        in the video without processing everything.
        """
        try:
            # Search vector database
            results = vector_db.search_similar(query, n_results=3)
            
            if not results or not results['documents']:
                return "No relevant segments found for this query."
            
            # Format results
            output_lines = ["Found relevant video segments:"]
            
            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i]
                segment_id = results['ids'][0][i]
                
                output_lines.append(
                    f"\nSegment {i+1} (ID: {segment_id}):"
                )
                output_lines.append(f"  Time: {meta['start_time']}s - {meta['end_time']}s")
                output_lines.append(f"  Description: {doc[:200]}...")
            
            return "\n".join(output_lines)
            
        except Exception as e:
            logger.error(f"Error in search_video_segments_tool: {e}")
            return f"Error searching segments: {str(e)}"
    
    def _get_pose_analysis_tool(self, segment_id: str) -> str:
        """
        Tool: Get detailed pose analysis for a segment.
        
        This implements the perception phase - extracting quantitative
        measurements from the video.
        """
        try:
            # Get segment metadata from vector DB
            result = vector_db.collection.get(ids=[segment_id])
            
            if not result or not result['metadatas']:
                return f"Segment {segment_id} not found."
            
            meta = result['metadatas'][0]
            chunk_path = meta.get('chunk_path')
            
            if not chunk_path or not Path(chunk_path).exists():
                return f"Video file not found for segment {segment_id}."
            
            # Extract pose data from this segment
            logger.info(f"Extracting pose data from {chunk_path}")
            pose_sequence = self.pose_estimator.extract_pose_from_video(
                chunk_path,
                sample_rate=3
            )
            
            if not pose_sequence:
                return f"No pose detected in segment {segment_id}."
            
            # Analyze the pose sequence
            pose_estimator = PoseEstimator()
            motion_stats = pose_estimator.analyze_motion_sequence(pose_sequence)
            
            # Format output
            output_lines = [
                f"Pose Analysis for Segment {segment_id}:",
                f"Time Range: {meta['start_time']}s - {meta['end_time']}s",
                f"Poses Detected: {len(pose_sequence)}",
                f"Average Confidence: {motion_stats.get('avg_confidence', 0):.1%}",
                "",
                "Joint Angle Ranges:"
            ]
            
            for angle_name, stats in motion_stats.get('angle_ranges', {}).items():
                output_lines.append(
                    f"  {angle_name}: {stats['min']:.1f}° - {stats['max']:.1f}° "
                    f"(mean: {stats['mean']:.1f}°)"
                )
            
            return "\n".join(output_lines)
            
        except Exception as e:
            logger.error(f"Error in get_pose_analysis_tool: {e}")
            return f"Error analyzing pose: {str(e)}"
    
    def _validate_safety_tool(self, input_str: str) -> str:
        """
        Tool: Validate motion against safety standards.
        
        This implements the reasoning phase - comparing measurements
        against established biomechanics guidelines.
        """
        try:
            # Parse input: "exercise_type|segment_id"
            parts = input_str.split('|')
            if len(parts) != 2:
                return "Invalid input format. Use: exercise_type|segment_id"
            
            exercise_type, segment_id = parts
            
            # Get segment
            result = vector_db.collection.get(ids=[segment_id])
            if not result or not result['metadatas']:
                return f"Segment {segment_id} not found."
            
            meta = result['metadatas'][0]
            chunk_path = meta.get('chunk_path')
            
            if not chunk_path or not Path(chunk_path).exists():
                return f"Video file not found for segment {segment_id}."
            
            # Extract pose data
            pose_sequence = self.pose_estimator.extract_pose_from_video(
                chunk_path,
                sample_rate=3
            )
            
            if not pose_sequence:
                return f"No pose detected in segment {segment_id}."
            
            # Analyze against guidelines
            analysis = self.motion_analyzer.analyze_pose_sequence(
                pose_sequence,
                exercise_type
            )
            
            # Format as structured report
            report = self.motion_analyzer.format_analysis_for_llm(analysis)
            
            return report
            
        except Exception as e:
            logger.error(f"Error in validate_safety_tool: {e}")
            return f"Error validating safety: {str(e)}"
    
    def _get_standards_tool(self, exercise_type: str) -> str:
        """
        Tool: Get biomechanics standards for an exercise.
        
        This provides the knowledge base of what "correct" looks like.
        """
        try:
            guidelines = self.motion_analyzer.guidelines.get('exercises', {})
            exercise_data = guidelines.get(exercise_type.lower())
            
            if not exercise_data:
                return f"No guidelines found for exercise: {exercise_type}"
            
            # Format output
            output_lines = [
                f"Biomechanics Standards for {exercise_data.get('name', exercise_type)}:",
                f"Description: {exercise_data.get('description', 'N/A')}",
                "",
                "Safety Standards:"
            ]
            
            for standard_name, standard in exercise_data.get('safety_standards', {}).items():
                output_lines.append(f"\n{standard_name.replace('_', ' ').title()}:")
                
                if 'safe_range' in standard:
                    output_lines.append(
                        f"  Safe Range: {standard['safe_range'][0]}-{standard['safe_range'][1]} {standard.get('unit', '')}"
                    )
                if 'measurement' in standard:
                    output_lines.append(f"  Measurement: {standard['measurement']}")
                if 'violation_risk' in standard:
                    output_lines.append(f"  Risk Level: {standard['violation_risk'].upper()}")
                if 'source' in standard:
                    output_lines.append(f"  Source: {standard['source']}")
            
            return "\n".join(output_lines)
            
        except Exception as e:
            logger.error(f"Error in get_standards_tool: {e}")
            return f"Error getting standards: {str(e)}"
    
    def answer_query(self, query: str) -> Dict[str, any]:
        """
        Main method: Answer a user query about video motion.
        
        This triggers the agentic workflow:
        1. Agent receives query
        2. Agent plans which tools to use
        3. Agent executes tools in sequence
        4. Agent synthesizes final answer with evidence
        
        Args:
            query: Natural language question about the video
            
        Returns:
            Dict with answer and evidence trail
        """
        try:
            logger.info(f"Processing query: {query}")
            
            # Execute agent
            result = self.executor.invoke({"input": query})
            
            # Extract answer and intermediate steps
            answer = result.get('output', 'Unable to generate answer.')
            
            # The intermediate_steps contain the tool calls and observations
            evidence = []
            if 'intermediate_steps' in result:
                for step in result['intermediate_steps']:
                    action, observation = step
                    evidence.append({
                        'tool': action.tool,
                        'input': action.tool_input,
                        'observation': observation[:500]  # Truncate long observations
                    })
            
            return {
                'answer': answer,
                'evidence': evidence,
                'query': query
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'answer': f"I encountered an error processing your query: {str(e)}",
                'evidence': [],
                'query': query
            }


# Global instance
agent_service = MotionAnalysisAgent()