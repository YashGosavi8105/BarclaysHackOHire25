import os
import re
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
from groq import Groq
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

class TermSheetExtractor:
    def __init__(self, 
                 embedding_model_name="all-MiniLM-L6-v2", 
                 llm_api_key=None,
                 llm_model="llama2-70b-4096"):
        """
        Initialize the extractor with a sentence transformer model for creating embeddings
        and LLM for extracting important points from any term sheet.
        
        Args:
            embedding_model_name: The sentence transformer model for creating embeddings
            llm_api_key: API key for LLM (Groq in this case)
            llm_model: The model to use for extraction
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.llm_api_key = llm_api_key or os.environ.get("Groq_API_KEY")
        self.llm_model = llm_model
        self.llm_client = Groq(api_key=self.llm_api_key) if self.llm_api_key else None
        
        self.document_text = None
        self.document_sections = {}
        self.validation_points = {}
        self.document_type = None
        self.document_metadata = {}
        
    def _initialize_llm_client(self):
        """Initialize the LLM client if not already initialized."""
        if not self.llm_client and self.llm_api_key:
            self.llm_client = Groq(api_key=self.llm_api_key)
        elif not self.llm_api_key:
            raise ValueError("LLM API key is required. Please provide via init or set GROQ_API_KEY environment variable.")
            
    def extract_from_document(self, document_text: str, document_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract important points from any term sheet document using an LLM.
        
        Args:
            document_text: The full term sheet text
            document_type: Optional hint about document type (loan, contract, etc.)
            
        Returns:
            Dict containing extracted structured validation points
        """
        self._initialize_llm_client()
        self.document_text = document_text
        
        # Determine document type if not provided
        if not document_type:
            document_type = self._detect_document_type(document_text)
        self.document_type = document_type
        
        # Extract document metadata
        self.document_metadata = self._extract_document_metadata(document_text)
        
        # Extract document sections
        self.document_sections = self._extract_sections(document_text)
        
        # Use LLM to identify important validation points
        self.validation_points = self._extract_validation_points_with_llm(document_text, document_type)
        
        return self.validation_points
    
    def _detect_document_type(self, document_text: str) -> str:
        """
        Use LLM to determine the type of term sheet document.
        
        Args:
            document_text: The document text
            
        Returns:
            Document type as a string
        """
        prompt = f"""
        Analyze the following document and determine its type (e.g., mortgage term sheet, 
        business loan agreement, service contract, etc.). Provide a concise document type name.
        
        DOCUMENT:
        {document_text[:4000]}  # Limit to first 4000 chars to stay within token limits
        
        OUTPUT ONLY THE DOCUMENT TYPE AS A SHORT STRING (e.g., "mortgage_term_sheet", "business_loan", "service_contract"):
        """
        
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "You analyze documents and determine their type. Respond with only the document type as a short string with underscores between words."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=50
        )
        
        document_type = response.choices[0].message.content.strip().lower()
        # Clean up response in case LLM includes explanations
        document_type = document_type.split("\n")[0].strip()
        return document_type
    
    def _extract_document_metadata(self, document_text: str) -> Dict[str, Any]:
        """
        Extract basic metadata about the document using LLM.
        
        Args:
            document_text: The document text
            
        Returns:
            Dictionary of metadata
        """
        prompt = f"""
        Extract basic metadata from the following document. Include:
        1. Document title
        2. Issuing organization/entity
        3. Date (if present)
        4. Version or revision number (if present)
        
        DOCUMENT:
        {document_text[:4000]}  # Limit to first 4000 chars
        
        Return the metadata as JSON with keys: title, issuer, date, version.
        """
        
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "You extract document metadata and return it as JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=200
        )
        
        metadata_text = response.choices[0].message.content
        
        # Extract JSON from response
        try:
            # Find JSON pattern in response
            json_match = re.search(r'{.*}', metadata_text, re.DOTALL)
            if json_match:
                metadata = json.loads(json_match.group(0))
            else:
                # If no JSON format found, create basic structure
                metadata = {
                    "title": "Unknown",
                    "issuer": "Unknown",
                    "date": None,
                    "version": None
                }
        except json.JSONDecodeError:
            # If JSON parsing fails, create basic structure
            metadata = {
                "title": "Unknown",
                "issuer": "Unknown",
                "date": None,
                "version": None
            }
            
        return metadata
    
    def _extract_sections(self, document_text: str) -> Dict[str, str]:
        """
        Extract main sections from the document text.
        
        Args:
            document_text: The document text
            
        Returns:
            Dictionary of section names to section content
        """
        # Use LLM to identify main sections in the document
        prompt = f"""
        Identify the main sections in this term sheet document. For each section, provide:
        1. The section name/title
        2. The section number (if any)
        
        DOCUMENT:
        {document_text[:8000]}  # Using more text for section identification
        
        Return the list of sections as JSON array with objects containing keys: name, number.
        """
        
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "You identify document sections and return them as JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        sections_text = response.choices[0].message.content
        
        # Extract sections from response
        sections = {}
        try:
            # Find JSON pattern in response
            json_match = re.search(r'\[.*\]', sections_text, re.DOTALL)
            if json_match:
                sections_list = json.loads(json_match.group(0))
                
                # Now extract content for each identified section
                for section_info in sections_list:
                    section_name = section_info.get("name", "").strip()
                    if not section_name:
                        continue
                        
                    # Try to find section content using section name as marker
                    pattern = f"(?:{section_name}|{section_info.get('number', '')}).*?(?=(?:{self._build_section_boundary_pattern(sections_list)})|$)"
                    section_match = re.search(pattern, document_text, re.DOTALL | re.IGNORECASE)
                    
                    if section_match:
                        sections[section_name] = section_match.group(0).strip()
                    else:
                        # Fallback: use LLM to extract this specific section
                        sections[section_name] = self._extract_specific_section(document_text, section_name)
            
        except (json.JSONDecodeError, AttributeError) as e:
            # If extraction fails, use regex as fallback
            section_pattern = r'(?:\*\*|\#\#\#?|\d+\.)(.*?)(?=(?:\*\*|\#\#\#?|\d+\.)|$)'
            section_matches = re.finditer(section_pattern, document_text)
            for match in section_matches:
                section_name = match.group(1).strip()
                if section_name:
                    sections[section_name] = match.group(0).strip()
                    
        return sections
    
    def _build_section_boundary_pattern(self, sections_list):
        """Build regex pattern from section names to identify section boundaries."""
        section_names = [re.escape(section.get("name", "")) for section in sections_list if section.get("name")]
        section_numbers = [re.escape(section.get("number", "")) for section in sections_list if section.get("number")]
        
        patterns = section_names + section_numbers
        return "|".join(patterns)
    
    def _extract_specific_section(self, document_text, section_name):
        """Use LLM to extract a specific section from the document."""
        prompt = f"""
        Extract the content for the section named "{section_name}" from this document.
        
        DOCUMENT:
        {document_text[:8000]}
        
        Return only the content of the section named "{section_name}".
        """
        
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": f"You extract the content for a section named '{section_name}' from a document."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    
    def _extract_validation_points_with_llm(self, document_text: str, document_type: str) -> Dict[str, Any]:
        """
        Use LLM to extract important validation points from the document.
        
        Args:
            document_text: The document text
            document_type: The type of document
            
        Returns:
            Dictionary of validation points by category
        """
        # Prepare system prompt based on document type
        system_prompt = f"""
        You are an expert in analyzing {document_type} documents. Your task is to extract 
        all the important validation points that would be used to verify compliance with 
        the term sheet. Focus on quantifiable requirements, thresholds, limits, eligibility 
        criteria, and conditions.
        
        Group the validation points into meaningful categories. For each validation point:
        1. Be specific and precise with values, rates, and thresholds
        2. Include any relevant formulas or calculations
        3. Note any conditional requirements
        
        Format your response as a JSON object where:
        - Each key is a category name
        - Each value is either an array of validation points or a nested object of key-value pairs
        
        For example (format will vary based on document content):
        {
          "eligibility_requirements": [
            "Must be X",
            "Must have Y",
            "Cannot have Z"
          ],
          "financial_thresholds": {
            "min_income": 50000,
            "max_debt_ratio": 0.43,
            "interest_rate": "5.25%"
          }
        }
        """
        
        # User prompt with document text
        user_prompt = f"""
        Extract all important validation points from this {document_type} document:
        
        {document_text[:12000]}  # Using more text for comprehensive extraction
        
        Return only the JSON with validation points grouped by category.
        """
        
        # Make request to LLM
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        response_text = response.choices[0].message.content
        
        # Extract JSON from response
        try:
            # Find JSON pattern in response
            json_match = re.search(r'{.*}', response_text, re.DOTALL)
            if json_match:
                validation_points = json.loads(json_match.group(0))
            else:
                # If no JSON format found, try parsing the entire response
                validation_points = json.loads(response_text)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract in a different way
            validation_points = self._extract_validation_points_fallback(document_text)
            
        return validation_points
    
    def _extract_validation_points_fallback(self, document_text: str) -> Dict[str, Any]:
        """Fallback method to extract validation points if JSON parsing fails."""
        # Simplified prompt requesting markdown format instead of JSON
        prompt = f"""
        Extract important validation points from this document, organizing them by category.
        Format your response as a markdown document with categories as headings.
        
        DOCUMENT:
        {document_text[:8000]}
        """
        
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "You extract important validation points from documents and format them as markdown with categories."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        
        markdown_response = response.choices[0].message.content
        
        # Parse markdown into categories and points
        validation_points = {}
        current_category = "general"
        validation_points[current_category] = []
        
        for line in markdown_response.split('\n'):
            if line.startswith('# ') or line.startswith('## '):
                # New category
                current_category = line.strip('# ').strip()
                validation_points[current_category] = []
            elif line.startswith('- ') or line.startswith('* '):
                # Validation point
                point = line.strip('- ').strip('* ').strip()
                if point:
                    validation_points[current_category].append(point)
                    
        return validation_points
    
    def create_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Create embeddings for all validation points for semantic search.
        
        Returns:
            Dictionary of category -> embeddings
        """
        embeddings = {}
        
        for category, data in self.validation_points.items():
            if isinstance(data, list):
                if data:  # Only process non-empty lists
                    # Convert list items to strings
                    texts = [str(item) for item in data]
                    category_embeddings = self.embedding_model.encode(texts)
                    embeddings[category] = category_embeddings
            elif isinstance(data, dict):
                # For dictionaries, encode the keys and values as text
                texts = [f"{key}: {value}" for key, value in data.items()]
                if texts:  # Only process if we have texts
                    category_embeddings = self.embedding_model.encode(texts)
                    embeddings[category] = category_embeddings
            
        return embeddings
    
    def save_data(self, output_dir="./termsheet_data"):
        """
        Save the extracted validation points and embeddings to files.
        
        Args:
            output_dir: Directory to save the data
        """
        # Create unique subfolder for this document
        if self.document_metadata.get("title", "").strip():
            doc_folder = self.document_metadata["title"].lower().replace(" ", "_")
        else:
            doc_folder = f"{self.document_type}_{int(time.time())}"
            
        output_path = Path(output_dir) / doc_folder
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save document metadata
        with open(output_path / "metadata.json", "w") as f:
            json.dump({
                "document_type": self.document_type,
                "metadata": self.document_metadata,
                "extraction_date": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)
            
        # Save validation points as JSON
        with open(output_path / "validation_points.json", "w") as f:
            json.dump(self.validation_points, f, indent=2)
            
        # Create and save embeddings
        embeddings = self.create_embeddings()
        with open(output_path / "embeddings.pkl", "wb") as f:
            pickle.dump(embeddings, f)
            
        # Create a simple lookup index
        lookup_index = {}
        for category, data in self.validation_points.items():
            if isinstance(data, list):
                lookup_index[category] = {i: item for i, item in enumerate(data)}
            elif isinstance(data, dict):
                lookup_index[category] = data
                
        with open(output_path / "lookup_index.json", "w") as f:
            json.dump(lookup_index, f, indent=2)
            
        print(f"Data saved to {output_path}/")
        return str(output_path)


class TermSheetValidator:
    def __init__(self, data_dir: str, embedding_model_name="all-MiniLM-L6-v2"):
        """
        Initialize validator with saved embeddings and data.
        
        Args:
            data_dir: Directory containing saved data
            embedding_model_name: Must match the model used for extraction
        """
        self.model = SentenceTransformer(embedding_model_name)
        self.data_dir = Path(data_dir)
        
        # Load saved data
        with open(self.data_dir / "metadata.json", "r") as f:
            self.metadata = json.load(f)
            
        with open(self.data_dir / "validation_points.json", "r") as f:
            self.validation_points = json.load(f)
            
        with open(self.data_dir / "embeddings.pkl", "rb") as f:
            self.embeddings = pickle.load(f)
            
        with open(self.data_dir / "lookup_index.json", "r") as f:
            self.lookup_index = json.load(f)
    
    def search_requirements(self, query: str, category: str = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant requirements using semantic similarity.
        
        Args:
            query: The search query
            category: Specific category to search in (optional)
            top_k: Number of results to return
            
        Returns:
            List of matched requirements with scores
        """
        query_embedding = self.model.encode([query])[0]
        results = []
        
        categories_to_search = [category] if category else self.embeddings.keys()
        
        for category_name in categories_to_search:
            if category_name in self.embeddings:
                category_embeddings = self.embeddings[category_name]
                
                # Calculate cosine similarity
                similarities = np.dot(category_embeddings, query_embedding) / (
                    np.linalg.norm(category_embeddings, axis=1) * np.linalg.norm(query_embedding)
                )
                
                # Get top matches
                best_indices = np.argsort(-similarities)[:top_k]
                
                for idx in best_indices:
                    # Get the item from lookup index
                    item = self.lookup_index[category_name].get(str(idx), 
                           self.lookup_index[category_name].get(idx, f"Item {idx} not found"))
                    
                    results.append({
                        "category": category_name,
                        "item": item,
                        "score": float(similarities[idx])
                    })
        
        # Sort by similarity score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def validate_data(self, input_data: Dict[str, Any], 
                       custom_validation_rules: Dict[str, callable] = None) -> Dict[str, Any]:
        """
        Validate input data against the term sheet requirements.
        
        Args:
            input_data: Dictionary containing data to validate
            custom_validation_rules: Optional dictionary of custom validation functions
            
        Returns:
            Validation results with pass/fail status
        """
        validation_results = {
            "overall_status": "PENDING",
            "categories": {},
            "document_type": self.metadata["document_type"],
            "document_title": self.metadata["metadata"].get("title", "Unknown")
        }
        
        # Apply default validation rules based on document type
        self._apply_default_validation(input_data, validation_results)
        
        # Apply any custom validation rules
        if custom_validation_rules:
            for rule_name, rule_func in custom_validation_rules.items():
                try:
                    rule_result = rule_func(input_data, self.validation_points)
                    validation_results["categories"][rule_name] = rule_result
                except Exception as e:
                    validation_results["categories"][rule_name] = {
                        "status": "ERROR",
                        "details": f"Error applying rule: {str(e)}"
                    }
        
        # Set overall status
        if any(cat.get("status") == "FAIL" for cat in validation_results["categories"].values()):
            validation_results["overall_status"] = "FAIL"
        elif len(validation_results["categories"]) > 0 and all(
            cat.get("status") in ["PASS", "WARNING"] for cat in validation_results["categories"].values()
        ):
            validation_results["overall_status"] = "PASS"
            
        return validation_results
    
    def _apply_default_validation(self, input_data: Dict[str, Any], 
                                  validation_results: Dict[str, Any]) -> None:
        """Apply basic validation rules based on document type."""
        document_type = self.metadata["document_type"]
        
        if "loan" in document_type.lower() or "mortgage" in document_type.lower():
            # Apply loan-related validations
            self._validate_numeric_thresholds(input_data, validation_results)
            
        elif "agreement" in document_type.lower() or "contract" in document_type.lower():
            # Apply agreement-related validations
            self._validate_requirements_list(input_data, validation_results)
            
        # Add general validations for all document types
        self._validate_dates(input_data, validation_results)
    
    def _validate_numeric_thresholds(self, input_data: Dict[str, Any], 
                                    validation_results: Dict[str, Any]) -> None:
        """Validate numeric thresholds from validation points."""
        # Look for categories that might contain numeric thresholds
        threshold_categories = [cat for cat in self.validation_points.keys() 
                               if any(kw in cat.lower() for kw in ["limit", "threshold", "requirement", "financial"])]
        
        for category in threshold_categories:
            if isinstance(self.validation_points[category], dict):
                # Process dictionary-style thresholds
                for key, threshold in self.validation_points[category].items():
                    if key in input_data:
                        # Try to convert threshold to numeric for comparison
                        try:
                            # Handle percentage values
                            if isinstance(threshold, str) and "%" in threshold:
                                threshold_value = float(threshold.replace("%", "")) / 100
                                input_value = float(input_data[key])
                                
                                if "max" in key.lower() or "limit" in key.lower():
                                    passed = input_value <= threshold_value
                                elif "min" in key.lower():
                                    passed = input_value >= threshold_value
                                else:
                                    # Default to equality check
                                    passed = abs(input_value - threshold_value) < 0.0001
                            else:
                                # Handle regular numeric values
                                threshold_value = float(threshold) if isinstance(threshold, str) else threshold
                                input_value = float(input_data[key])
                                
                                if "max" in key.lower() or "limit" in key.lower():
                                    passed = input_value <= threshold_value
                                elif "min" in key.lower():
                                    passed = input_value >= threshold_value
                                else:
                                    # Default to equality check
                                    passed = abs(input_value - threshold_value) < 0.0001
                                    
                            # Add result
                            if category not in validation_results["categories"]:
                                validation_results["categories"][category] = {
                                    "status": "PASS" if passed else "FAIL",
                                    "details": {}
                                }
                            
                            validation_results["categories"][category]["details"][key] = {
                                "status": "PASS" if passed else "FAIL",
                                "input": input_value,
                                "threshold": threshold_value,
                                "comparison": "≤" if "max" in key.lower() else "≥" if "min" in key.lower() else "="
                            }
                            
                            # Update category status if any check fails
                            if not passed and validation_results["categories"][category]["status"] != "FAIL":
                                validation_results["categories"][category]["status"] = "FAIL"
                                
                        except (ValueError, TypeError):
                            # Skip if conversion fails
                            continue
    
    def _validate_requirements_list(self, input_data: Dict[str, Any], 
                                   validation_results: Dict[str, Any]) -> None:
        """Validate against requirement lists from validation points."""
        # Look for categories that might contain requirement lists
        req_categories = [cat for cat in self.validation_points.keys() 
                         if any(kw in cat.lower() for kw in ["requirement", "eligibility", "condition", "obligation"])]
        
        for category in req_categories:
            if isinstance(self.validation_points[category], list):
                # Process list-style requirements
                
                # Create a requirements checklist
                requirements_status = {}
                for i, req in enumerate(self.validation_points[category]):
                    # Look for a matching key in input data based on some simple heuristics
                    # This is where more sophisticated matching logic would be implemented in a real system
                    found = False
                    for key, value in input_data.items():
                        # Try naive keyword matching
                        req_lower = req.lower()
                        key_lower = key.lower()
                        
                        if (key_lower in req_lower or any(word in req_lower for word in key_lower.split('_'))):
                            # Found potential match, check if value indicates compliance
                            if isinstance(value, bool):
                                requirements_status[req] = {
                                    "status": "PASS" if value else "FAIL",
                                    "input": value
                                }
                            else:
                                # For non-boolean values, assume presence indicates compliance
                                requirements_status[req] = {
                                    "status": "PASS",
                                    "input": value
                                }
                            found = True
                            break
                    
                    if not found:
                        # If no matching key found, mark as unknown
                        requirements_status[req] = {
                            "status": "UNKNOWN",
                            "details": "No matching data provided"
                        }
                
                # Add results to validation_results
                validation_results["categories"][category] = {
                    "status": "PASS" if all(r["status"] == "PASS" for r in requirements_status.values()) else "FAIL",
                    "details": requirements_status
                }
    
    def _validate_dates(self, input_data: Dict[str, Any], 
                        validation_results: Dict[str, Any]) -> None:
        """Validate date-related fields."""
        # Look for date-related fields in input data
        date_fields = [key for key in input_data.keys() 
                       if any(kw in key.lower() for kw in ["date", "deadline", "expiration", "start", "end"])]
        
        if date_fields:
            date_validation = {}
            
            for field in date_fields:
                # This is a simplified validation - in a real system you'd want more robust date parsing
                date_validation[field] = {
                    "status": "PASS",  # Default to pass, would actually compare against term sheet dates
                    "input": input_data[field]
                }
            
            validation_results["categories"]["dates"] = {
                "status": "PASS",
                "details": date_validation
            }


class TermSheetManager:
    def __init__(self, base_dir="./termsheet_data", embedding_model_name="all-MiniLM-L6-v2"):
        """
        Manager for multiple term sheets.
        
        Args:
            base_dir: Base directory for all term sheet data
            embedding_model_name: Model to use for embeddings
        """
        self.base_dir = Path(base_dir)
        self.embedding_model_name = embedding_model_name
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Load index of available term sheets
        self.term_sheets = self._load_term_sheet_index()
    
    def _load_term_sheet_index(self) -> Dict[str, Dict[str, Any]]:
        """Load index of available term sheets from base directory."""
        term_sheets = {}
        
        # Scan for term sheet directories
        for path in self.base_dir.iterdir():
            if path.is_dir() and (path / "metadata.json").exists():
                try:
                    with open(path / "metadata.json", "r") as f:
                        metadata = json.load(f)
                        
                    term_sheets[path.name] = {
                        "path": str(path),
                        "document_type": metadata["document_type"],
                        "title": metadata["metadata"].get("title", "Unknown"),
                        "issuer": metadata["metadata"].get("issuer", "Unknown"),
                        "date": metadata["metadata"].get("date", "Unknown"),
                        "extraction_date": metadata.get("extraction_date", "Unknown")
                    }
                except (json.JSONDecodeError, KeyError):
                    # Skip if metadata can't be read
                    continue
                    
        return term_sheets
    