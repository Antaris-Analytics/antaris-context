"""
Main context manager for coordinating context window optimization.
"""

from typing import Dict, List, Optional, Any, Union
import json
import os
import time
from .window import ContextWindow
from .compressor import MessageCompressor
from .strategy import ContextStrategy, RecencyStrategy, RelevanceStrategy, HybridStrategy, BudgetStrategy
from .profiler import ContextProfiler
from .utils import atomic_write_json


class ContextManager:
    """Main class for managing context windows, budgets, and optimization strategies."""
    
    # Section templates for common agent patterns
    SECTION_TEMPLATES = {
        'chatbot': {
            'system': 800,    # Basic system prompt
            'memory': 1500,   # Recent conversation memory
            'conversation': 5000,  # Active conversation
            'tools': 700     # Tool definitions
        },
        'agent_with_tools': {
            'system': 1200,   # System prompt + agent instructions
            'memory': 2000,   # Working memory and state
            'conversation': 3500,  # User interactions
            'tools': 1300     # Tool definitions and examples
        },
        'rag_pipeline': {
            'system': 600,    # Simple system prompt
            'memory': 1000,   # Search queries and context
            'conversation': 4500,  # Q&A conversation
            'tools': 1900     # RAG tools and retrieval results  
        },
        'code_assistant': {
            'system': 1000,   # Coding guidelines and style
            'memory': 1800,   # Recent code context
            'conversation': 4000,  # Code discussion
            'tools': 1200     # Code analysis and formatting tools
        },
        'balanced': {
            'system': 1000,   # Equal-ish distribution
            'memory': 2000,   
            'conversation': 4000,
            'tools': 1000
        }
    }
    
    def __init__(self, total_budget: int = 8000, config_file: Optional[str] = None, template: Optional[str] = None):
        """Initialize context manager.
        
        Args:
            total_budget: Total token budget for context window
            config_file: Optional JSON configuration file path
            template: Optional section template ('chatbot', 'agent_with_tools', 'rag_pipeline', etc.)
        """
        self.total_budget = total_budget
        self.config_file = config_file
        self.window = ContextWindow(total_budget)
        self.compressor = MessageCompressor('moderate')
        self.strategy = HybridStrategy()  # Default strategy
        self.profiler = ContextProfiler()
        
        # Default configuration - apply template if provided
        default_budgets = self.SECTION_TEMPLATES.get(template, {
            'system': 1000,
            'memory': 2000,
            'conversation': 4000,
            'tools': 1000
        })
        
        self.config = {
            'compression_level': 'moderate',
            'strategy': 'hybrid',
            'strategy_params': {
                'recency_weight': 0.4,
                'relevance_weight': 0.6
            },
            'section_budgets': default_budgets.copy(),
            'truncation_strategy': 'oldest_first',
            'auto_compress': True,
            'profiler_log_file': None,
            'template': template,  # Store template for reference
            'adaptive_budgets': {
                'enabled': False,
                'usage_history': [],
                'reallocation_threshold': 0.3
            }
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
            atomic_write_json(save_path, self.config, indent=2)
        except Exception as e:
            raise ValueError(f"Failed to save configuration to {save_path}: {e}")
    
    def apply_template(self, template_name: str) -> None:
        """Apply a section template to current configuration.
        
        Args:
            template_name: Template name ('chatbot', 'agent_with_tools', etc.)
        """
        if template_name not in self.SECTION_TEMPLATES:
            available = list(self.SECTION_TEMPLATES.keys())
            raise ValueError(f"Unknown template '{template_name}'. Available: {available}")
        
        template = self.SECTION_TEMPLATES[template_name]
        for section, budget in template.items():
            self.set_section_budget(section, budget)
        
        self.config['template'] = template_name
    
    @classmethod
    def get_available_templates(cls) -> Dict[str, Dict[str, int]]:
        """Get all available section templates."""
        return cls.SECTION_TEMPLATES.copy()
    
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
        
        # Collect all content items with metadata
        all_items = []
        for section_name, section_data in self.window.sections.items():
            for item in section_data['content']:
                all_items.append({
                    'section': section_name,
                    'item': item,
                    'added_at': item.get('added_at', 0),
                    'tokens': item.get('tokens', 0),
                    'priority': item.get('priority', 'normal')
                })
        
        # Sort by age (oldest first), but preserve critical priority items
        all_items.sort(key=lambda x: (
            x['priority'] == 'critical',  # Critical items last
            -x['added_at']  # Then by age, oldest first
        ))
        
        # Mark items for removal instead of removing during iteration
        items_to_remove = []
        for item_data in all_items:
            if tokens_removed >= tokens_to_remove:
                break
            
            if item_data['priority'] != 'critical':  # Never remove critical items
                items_to_remove.append(item_data)
                tokens_removed += item_data['tokens']
        
        # Remove items by rebuilding section content lists
        for section_name, section_data in self.window.sections.items():
            new_content = []
            for item in section_data['content']:
                # Keep item if it's not marked for removal
                if not any(marked['item'] is item for marked in items_to_remove):
                    new_content.append(item)
                else:
                    section_data['used'] -= item.get('tokens', 0)
            section_data['content'] = new_content
        
        return tokens_removed
    
    def _truncate_lowest_priority(self, tokens_to_remove: int) -> int:
        """Remove lowest priority content first."""
        tokens_removed = 0
        priority_order = ['optional', 'normal', 'important']  # Never remove critical
        
        # Collect all removal candidates across sections
        candidates = []
        for priority in priority_order:
            for section_name, section_data in self.window.sections.items():
                for item in section_data['content']:
                    if item.get('priority', 'normal') == priority:
                        candidates.append({
                            'section': section_name,
                            'item': item,
                            'priority': priority,
                            'tokens': item.get('tokens', 0)
                        })
        
        # Mark items for removal until budget met
        items_to_remove = []
        for candidate in candidates:
            if tokens_removed >= tokens_to_remove:
                break
            items_to_remove.append(candidate)
            tokens_removed += candidate['tokens']
        
        # Rebuild section content lists in a single pass
        remove_set = {id(c['item']) for c in items_to_remove}
        for section_name, section_data in self.window.sections.items():
            new_content = []
            for item in section_data['content']:
                if id(item) in remove_set:
                    section_data['used'] -= item.get('tokens', 0)
                else:
                    new_content.append(item)
            section_data['content'] = new_content
        
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
    
    def enable_adaptive_budgets(self, enabled: bool = True, reallocation_threshold: float = 0.3) -> None:
        """Enable or disable adaptive budget management.
        
        Args:
            enabled: Whether to enable adaptive budgets
            reallocation_threshold: Threshold for triggering reallocation (0.0-1.0)
        """
        self.config['adaptive_budgets']['enabled'] = enabled
        self.config['adaptive_budgets']['reallocation_threshold'] = reallocation_threshold
    
    def track_usage(self) -> None:
        """Track current usage for adaptive budget analysis."""
        if not self.config['adaptive_budgets']['enabled']:
            return
            
        usage_snapshot = {
            'timestamp': time.time(),
            'total_used': self.window.get_total_used(),
            'section_usage': {},
            'section_utilization': {}
        }
        
        for section_name, section_data in self.window.sections.items():
            usage_snapshot['section_usage'][section_name] = section_data['used']
            usage_snapshot['section_utilization'][section_name] = self.window.get_section_utilization(section_name)
        
        # Keep only recent history (last 100 snapshots)
        history = self.config['adaptive_budgets']['usage_history']
        history.append(usage_snapshot)
        if len(history) > 100:
            history.pop(0)
    
    def suggest_adaptive_reallocation(self) -> Optional[Dict]:
        """Suggest budget reallocation based on usage history.
        
        Returns:
            Reallocation suggestions or None if not enough data
        """
        if not self.config['adaptive_budgets']['enabled']:
            return None
            
        history = self.config['adaptive_budgets']['usage_history']
        if len(history) < 10:  # Need at least 10 data points
            return None
        
        # Calculate average utilization for each section
        section_avg_utilization = {}
        for section in self.window.sections:
            utilizations = [snapshot['section_utilization'][section] for snapshot in history[-20:]]  # Last 20 snapshots
            section_avg_utilization[section] = sum(utilizations) / len(utilizations)
        
        # Identify over/under utilized sections
        threshold = self.config['adaptive_budgets']['reallocation_threshold']
        underutilized = {section: util for section, util in section_avg_utilization.items() if util < threshold}
        overutilized = {section: util for section, util in section_avg_utilization.items() if util > (1.0 - threshold)}
        
        if not underutilized and not overutilized:
            return None  # No reallocation needed
        
        # Calculate suggested new budgets
        current_budgets = {section: data['budget'] for section, data in self.window.sections.items()}
        suggested_budgets = current_budgets.copy()
        
        # Redistribute from underutilized to overutilized sections
        total_to_redistribute = 0
        for section, util in underutilized.items():
            reduction = int(current_budgets[section] * (threshold - util))
            suggested_budgets[section] -= reduction
            total_to_redistribute += reduction
        
        # Distribute to overutilized sections proportionally
        if total_to_redistribute > 0 and overutilized:
            overutil_total = sum(overutilized.values())
            for section, util in overutilized.items():
                proportion = util / overutil_total
                suggested_budgets[section] += int(total_to_redistribute * proportion)
        
        return {
            'current_budgets': current_budgets,
            'suggested_budgets': suggested_budgets,
            'underutilized': underutilized,
            'overutilized': overutilized,
            'potential_savings': sum(current_budgets[s] - suggested_budgets[s] for s in underutilized)
        }
    
    def apply_adaptive_reallocation(self, auto_apply: bool = False) -> bool:
        """Apply adaptive budget reallocation.
        
        Args:
            auto_apply: Apply suggestions automatically without confirmation
            
        Returns:
            True if reallocation was applied, False otherwise
        """
        suggestions = self.suggest_adaptive_reallocation()
        if not suggestions:
            return False
        
        if auto_apply:
            for section, budget in suggestions['suggested_budgets'].items():
                self.set_section_budget(section, budget)
            return True
        
        return False
    
    def cascade_overflow(self, source_section: str) -> int:
        """Reallocate unused budget slack from other sections to cover overflow.
        
        This transfers budget allocation (not content) from sections that have
        unused capacity to the overflowing section. Only takes from slack
        (budget - used), never displaces existing content.
        
        Args:
            source_section: Section that is over budget
            
        Returns:
            Amount of budget tokens successfully reallocated
        """
        if source_section not in self.window.sections:
            return 0
            
        source_data = self.window.sections[source_section]
        overflow = max(0, source_data['used'] - source_data['budget'])
        
        if overflow == 0:
            return 0
        
        # Find sections with available budget, prioritize by available space
        available_sections = []
        for section_name, section_data in self.window.sections.items():
            if section_name != source_section:
                available = section_data['budget'] - section_data['used']
                if available > 0:
                    available_sections.append((available, section_name))
        
        available_sections.sort(reverse=True)  # Most available first
        
        redistributed = 0
        for available, section_name in available_sections:
            if redistributed >= overflow:
                break
            
            # Take as much as possible from this section
            take_amount = min(available, overflow - redistributed)
            
            # Transfer budget (conceptually - adjust budgets)
            self.window.sections[section_name]['budget'] -= take_amount
            source_data['budget'] += take_amount
            redistributed += take_amount
        
        return redistributed
    
    def save_snapshot(self, name: str) -> None:
        """Save a structural snapshot of the current context state.
        
        Captures configuration, section budgets, usage counts, and strategy.
        Does NOT serialize section content — snapshots are lightweight
        structural checkpoints. After restore, section budgets and used
        counts are restored but content lists will be empty.
        
        Args:
            name: Name for the snapshot
        """
        if not hasattr(self, '_snapshots'):
            self._snapshots = {}
        
        snapshot = {
            'timestamp': time.time(),
            'config': json.loads(json.dumps(self.config)),  # Deep copy
            'window_state': json.loads(self.window.to_json()),
            'strategy_name': self.strategy.get_strategy_name(),
            'compression_stats': self.compressor.get_compression_stats()
        }
        
        self._snapshots[name] = snapshot
    
    def restore_snapshot(self, name: str) -> bool:
        """Restore structural state from a saved snapshot.
        
        Restores configuration, section budgets, and strategy settings.
        Content is NOT restored — only structural state is preserved in
        snapshots. After restore, used counts reflect the snapshot but
        content lists will be empty.
        
        Args:
            name: Name of the snapshot to restore
            
        Returns:
            True if snapshot was restored, False if not found
        """
        if not hasattr(self, '_snapshots') or name not in self._snapshots:
            return False
        
        snapshot = self._snapshots[name]
        
        # Restore configuration
        self.config = snapshot['config'].copy()
        
        # Restore window structure  
        self.window = ContextWindow.from_json(json.dumps(snapshot['window_state']))
        
        # Apply configuration
        self._apply_config()
        
        return True
    
    def list_snapshots(self) -> List[Dict]:
        """List all saved snapshots.
        
        Returns:
            List of snapshot info (name, timestamp)
        """
        if not hasattr(self, '_snapshots'):
            return []
        
        return [
            {
                'name': name,
                'timestamp': snapshot['timestamp'],
                'created': snapshot['timestamp']  # For compatibility
            }
            for name, snapshot in self._snapshots.items()
        ]