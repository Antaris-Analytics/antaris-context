"""
Main context manager for coordinating context window optimization.
"""

from typing import Dict, List, Optional, Any, Union
import json
import os
from .window import ContextWindow
from .compressor import MessageCompressor
from .strategy import ContextStrategy, RecencyStrategy, RelevanceStrategy, HybridStrategy, BudgetStrategy
from .profiler import ContextProfiler


class ContextManager:
    """Main class for managing context windows, budgets, and optimization strategies."""
    
    def __init__(self, total_budget: int = 8000, config_file: Optional[str] = None):
        """Initialize context manager.
        
        Args:
            total_budget: Total token budget for context window
            config_file: Optional JSON configuration file path
        """
        self.total_budget = total_budget
        self.config_file = config_file
        self.window = ContextWindow(total_budget)
        self.compressor = MessageCompressor('moderate')
        self.strategy = HybridStrategy()  # Default strategy
        self.profiler = ContextProfiler()
        
        # Default configuration
        self.config = {
            'compression_level': 'moderate',
            'strategy': 'hybrid',
            'strategy_params': {
                'recency_weight': 0.4,
                'relevance_weight': 0.6
            },
            'section_budgets': {
                'system': 1000,
                'memory': 2000,
                'conversation': 4000,
                'tools': 1000
            },
            'truncation_strategy': 'oldest_first',
            'auto_compress': True,
            'profiler_log_file': None
        }
        
        # Load configuration if file provided
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
        
        self._apply_config()
    
    def load_config(self, config_file: str) -> None:
        """Load configuration from JSON file.
        
        Args:
            config_file: Path to JSON configuration file
        """
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
            
            # Merge with defaults
            self.config.update(file_config)
            self.config_file = config_file
            
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {config_file}: {e}")
    
    def save_config(self, config_file: Optional[str] = None) -> None:
        """Save current configuration to JSON file.
        
        Args:
            config_file: Optional path to save config (uses self.config_file if not provided)
        """
        save_path = config_file or self.config_file
        if not save_path:
            raise ValueError("No configuration file path specified")
        
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            raise ValueError(f"Failed to save configuration to {save_path}: {e}")
    
    def set_section_budget(self, section: str, budget: int) -> None:
        """Set token budget for a specific section.
        
        Args:
            section: Section name (system, memory, conversation, tools)
            budget: Token budget for this section
        """
        self.window.set_section_budget(section, budget)
        self.config['section_budgets'][section] = budget
    
    def add_content(self, section: str, content: Union[str, List[Dict]], 
                   priority: str = 'normal', compress: Optional[bool] = None, query: Optional[str] = None) -> bool:
        """Add content to a section with optional compression and strategy application.
        
        Args:
            section: Target section
            content: Content to add (string or list of message dicts)
            priority: Priority level (critical, important, normal, optional)
            compress: Whether to compress content (uses auto_compress config if None)
            query: Optional query for relevance-based content selection
            
        Returns:
            True if all content fits within budget, False if truncation occurred
        """
        if compress is None:
            compress = self.config['auto_compress']
        
        # Handle different content types
        if isinstance(content, str):
            # Single string content
            processed_content = self.compressor.compress(content) if compress else content
            return self.window.add_content(section, processed_content, priority)
        
        elif isinstance(content, list):
            # List of messages - apply strategy selection
            if compress:
                content = self.compressor.compress_message_list(content)
            
            # Get current section budget and usage
            section_data = self.window.sections[section]
            available_budget = section_data['budget'] - section_data['used']
            
            if available_budget <= 0:
                return False
            
            # Convert messages to content items for strategy
            content_items = []
            for i, msg in enumerate(content):
                if isinstance(msg, dict):
                    msg_content = msg.get('content', str(msg))
                    tokens = self.window._estimate_tokens(msg_content)
                    content_items.append({
                        'content': msg_content,
                        'tokens': tokens,
                        'priority': msg.get('priority', priority),
                        'added_at': i,
                        'original_message': msg
                    })
            
            # Apply strategy to select content within budget
            selected_items = self.strategy.select_content(content_items, available_budget, query)
            
            # Add selected content to window
            all_fit = True
            for item in selected_items:
                if not self.window.add_content(section, item['content'], item['priority']):
                    all_fit = False
            
            return all_fit
        
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")
    
    def optimize_context(self, query: Optional[str] = None, target_utilization: float = 0.85) -> Dict:
        """Optimize current context window to achieve target utilization.
        
        Args:
            query: Optional query for relevance-based optimization
            target_utilization: Target utilization ratio (0.0 - 1.0)
            
        Returns:
            Optimization report
        """
        optimization_report = {
            'initial_state': self.get_usage_report(),
            'actions_taken': [],
            'final_state': {},
            'success': False
        }
        
        initial_utilization = self.window.get_total_used() / self.total_budget
        
        # If already at target, no optimization needed
        if abs(initial_utilization - target_utilization) < 0.05:
            optimization_report['success'] = True
            optimization_report['final_state'] = self.get_usage_report()
            return optimization_report
        
        actions_taken = []
        
        # Step 1: Apply compression if not already applied
        if not self.config['auto_compress']:
            self._apply_compression()
            actions_taken.append('Applied content compression')
        
        # Step 2: If over budget, apply truncation strategy
        if initial_utilization > target_utilization:
            truncated_tokens = self._apply_truncation(target_utilization)
            if truncated_tokens > 0:
                actions_taken.append(f'Truncated {truncated_tokens} tokens using {self.config["truncation_strategy"]} strategy')
        
        # Step 3: If under budget and we have more content, try to add more
        elif initial_utilization < target_utilization:
            # This would require access to additional content pool
            actions_taken.append('Context under-utilized but no additional content available')
        
        # Step 4: Re-balance section budgets if needed
        current_usage = self.window.get_total_used()
        if current_usage > 0:
            rebalance_suggestions = self.profiler.suggest_budget_reallocation(self.window)
            if rebalance_suggestions['potential_improvements']:
                actions_taken.append('Budget rebalancing suggestions available')
        
        final_utilization = self.window.get_total_used() / self.total_budget
        optimization_report['actions_taken'] = actions_taken
        optimization_report['final_state'] = self.get_usage_report()
        optimization_report['success'] = abs(final_utilization - target_utilization) < 0.1
        
        return optimization_report
    
    def set_strategy(self, strategy_name: str, **kwargs) -> None:
        """Set the content selection strategy.
        
        Args:
            strategy_name: Strategy name ('recency', 'relevance', 'hybrid', 'budget')
            **kwargs: Strategy-specific parameters
        """
        if strategy_name == 'recency':
            self.strategy = RecencyStrategy(**kwargs)
        elif strategy_name == 'relevance':
            self.strategy = RelevanceStrategy(**kwargs)
        elif strategy_name == 'hybrid':
            self.strategy = HybridStrategy(**kwargs)
        elif strategy_name == 'budget':
            self.strategy = BudgetStrategy(**kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        self.config['strategy'] = strategy_name
        self.config['strategy_params'] = kwargs
    
    def set_compression_level(self, level: str) -> None:
        """Set message compression level.
        
        Args:
            level: Compression level ('light', 'moderate', 'aggressive')
        """
        self.compressor = MessageCompressor(level)
        self.config['compression_level'] = level
    
    def get_usage_report(self) -> Dict:
        """Get comprehensive usage report."""
        base_report = self.window.get_usage_report()
        
        # Add strategy and compression info
        base_report['configuration'] = {
            'strategy': self.strategy.get_strategy_name(),
            'compression_level': self.config['compression_level'],
            'auto_compress': self.config['auto_compress'],
            'truncation_strategy': self.config['truncation_strategy']
        }
        
        # Add compression stats if available
        base_report['compression_stats'] = self.compressor.get_compression_stats()
        
        return base_report
    
    def analyze_context(self, log_analysis: bool = True) -> Dict:
        """Perform comprehensive context analysis using the profiler.
        
        Args:
            log_analysis: Whether to log the analysis results
            
        Returns:
            Analysis report
        """
        if log_analysis and self.config.get('profiler_log_file'):
            self.profiler.log_file = self.config['profiler_log_file']
        
        return self.profiler.analyze_window(self.window)
    
    def clear_section(self, section: str) -> None:
        """Clear all content from a section."""
        self.window.clear_section(section)
    
    def clear_all_content(self) -> None:
        """Clear all content from all sections."""
        for section in self.window.sections:
            self.window.clear_section(section)
    
    def get_section_content(self, section: str) -> List[Dict]:
        """Get all content items for a section."""
        return self.window.get_section_content(section)
    
    def is_over_budget(self) -> bool:
        """Check if context window is over budget."""
        return self.window.is_over_budget()
    
    def get_available_budget(self, section: Optional[str] = None) -> Union[int, Dict[str, int]]:
        """Get available budget for a section or all sections.
        
        Args:
            section: Specific section name, or None for all sections
            
        Returns:
            Available budget (int) or dict of all sections
        """
        if section:
            if section not in self.window.sections:
                raise ValueError(f"Unknown section: {section}")
            section_data = self.window.sections[section]
            return max(0, section_data['budget'] - section_data['used'])
        else:
            available = {}
            for section_name, section_data in self.window.sections.items():
                available[section_name] = max(0, section_data['budget'] - section_data['used'])
            return available
    
    def export_state(self) -> str:
        """Export current state to JSON string (structure only, no content)."""
        state = {
            'config': self.config,
            'window_state': json.loads(self.window.to_json()),
            'strategy_name': self.strategy.get_strategy_name(),
            'compression_stats': self.compressor.get_compression_stats()
        }
        return json.dumps(state, indent=2)
    
    def import_state(self, state_json: str) -> None:
        """Import state from JSON string (structure only)."""
        try:
            state = json.loads(state_json)
            
            # Restore configuration
            self.config.update(state.get('config', {}))
            
            # Restore window structure
            if 'window_state' in state:
                self.window = ContextWindow.from_json(json.dumps(state['window_state']))
            
            # Apply configuration
            self._apply_config()
            
        except Exception as e:
            raise ValueError(f"Failed to import state: {e}")
    
    def _apply_config(self) -> None:
        """Apply current configuration to components."""
        # Set section budgets
        for section, budget in self.config['section_budgets'].items():
            if section in self.window.sections:
                self.window.set_section_budget(section, budget)
        
        # Set compression level
        self.compressor = MessageCompressor(self.config['compression_level'])
        
        # Set strategy
        strategy_name = self.config['strategy']
        strategy_params = self.config.get('strategy_params', {})
        
        try:
            self.set_strategy(strategy_name, **strategy_params)
        except ValueError:
            # Fall back to default if strategy config is invalid
            self.strategy = HybridStrategy()
        
        # Set profiler log file
        if self.config.get('profiler_log_file'):
            self.profiler.log_file = self.config['profiler_log_file']
    
    def _apply_compression(self) -> int:
        """Apply compression to all existing content and return tokens saved."""
        tokens_saved = 0
        
        for section_name, section_data in self.window.sections.items():
            for item in section_data['content']:
                if 'content' in item:
                    original_content = item['content']
                    compressed_content = self.compressor.compress(original_content)
                    
                    if compressed_content != original_content:
                        original_tokens = item['tokens']
                        new_tokens = self.window._estimate_tokens(compressed_content)
                        tokens_saved += (original_tokens - new_tokens)
                        
                        # Update item
                        item['content'] = compressed_content
                        item['tokens'] = new_tokens
                        
            # Recalculate section usage
            section_data['used'] = sum(item.get('tokens', 0) for item in section_data['content'])
        
        return tokens_saved
    
    def _apply_truncation(self, target_utilization: float) -> int:
        """Apply truncation strategy to reach target utilization."""
        target_tokens = int(self.total_budget * target_utilization)
        current_tokens = self.window.get_total_used()
        
        if current_tokens <= target_tokens:
            return 0
        
        tokens_to_remove = current_tokens - target_tokens
        tokens_removed = 0
        
        strategy = self.config['truncation_strategy']
        
        if strategy == 'oldest_first':
            tokens_removed = self._truncate_oldest_first(tokens_to_remove)
        elif strategy == 'lowest_priority':
            tokens_removed = self._truncate_lowest_priority(tokens_to_remove)
        elif strategy == 'smart_summary_markers':
            tokens_removed = self._truncate_smart_summary(tokens_to_remove)
        else:
            # Default to oldest first
            tokens_removed = self._truncate_oldest_first(tokens_to_remove)
        
        return tokens_removed
    
    def _truncate_oldest_first(self, tokens_to_remove: int) -> int:
        """Remove oldest content first until target is reached."""
        tokens_removed = 0
        
        # Collect all content with age info
        all_content = []
        for section_name, section_data in self.window.sections.items():
            for i, item in enumerate(section_data['content']):
                all_content.append({
                    'section': section_name,
                    'index': i,
                    'added_at': item.get('added_at', 0),
                    'tokens': item.get('tokens', 0),
                    'priority': item.get('priority', 'normal')
                })
        
        # Sort by age (oldest first), but preserve critical priority items
        all_content.sort(key=lambda x: (
            x['priority'] == 'critical',  # Critical items last
            -x['added_at']  # Then by age, oldest first
        ))
        
        # Remove items until target reached
        for item in all_content:
            if tokens_removed >= tokens_to_remove:
                break
            
            if item['priority'] != 'critical':  # Never remove critical items
                section = self.window.sections[item['section']]
                if item['index'] < len(section['content']):
                    removed_item = section['content'].pop(item['index'])
                    removed_tokens = removed_item.get('tokens', 0)
                    section['used'] -= removed_tokens
                    tokens_removed += removed_tokens
                    
                    # Update indices for remaining items in this section
                    for other_item in all_content:
                        if (other_item['section'] == item['section'] and 
                            other_item['index'] > item['index']):
                            other_item['index'] -= 1
        
        return tokens_removed
    
    def _truncate_lowest_priority(self, tokens_to_remove: int) -> int:
        """Remove lowest priority content first."""
        tokens_removed = 0
        priority_order = ['optional', 'normal', 'important']  # Never remove critical
        
        for priority in priority_order:
            if tokens_removed >= tokens_to_remove:
                break
            
            for section_name, section_data in self.window.sections.items():
                items_to_remove = []
                for i, item in enumerate(section_data['content']):
                    if item.get('priority', 'normal') == priority:
                        items_to_remove.append((i, item))
                
                # Remove items (reverse order to maintain indices)
                for i, item in reversed(items_to_remove):
                    if tokens_removed >= tokens_to_remove:
                        break
                    
                    section_data['content'].pop(i)
                    removed_tokens = item.get('tokens', 0)
                    section_data['used'] -= removed_tokens
                    tokens_removed += removed_tokens
        
        return tokens_removed
    
    def _truncate_smart_summary(self, tokens_to_remove: int) -> int:
        """Smart truncation preserving summary markers and important structure."""
        # This is a simplified implementation
        # In practice, this would be more sophisticated
        tokens_removed = 0
        
        for section_name, section_data in self.window.sections.items():
            if tokens_removed >= tokens_to_remove:
                break
            
            items_to_process = []
            for i, item in enumerate(section_data['content']):
                content = item.get('content', '')
                # Skip items that look like summaries or headers
                if not (content.strip().startswith('#') or 
                       'summary' in content.lower() or
                       len(content.split()) < 10):  # Short items are likely important
                    items_to_process.append((i, item))
            
            # Remove from end of list to preserve indices
            for i, item in reversed(items_to_process):
                if tokens_removed >= tokens_to_remove:
                    break
                
                if item.get('priority', 'normal') != 'critical':
                    section_data['content'].pop(i)
                    removed_tokens = item.get('tokens', 0)
                    section_data['used'] -= removed_tokens
                    tokens_removed += removed_tokens
        
        return tokens_removed