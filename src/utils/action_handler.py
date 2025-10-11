"""
Action Handler for Smart Home Automation and Optimization
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime


class ActionHandler:
    """
    Handles actions based on detected behaviour and model predictions.
    Executes automation and optimization strategies.
    """
    
    def __init__(self, log_actions: bool = True):
        """
        Initialize the action handler.
        
        Args:
            log_actions: Whether to log all actions taken
        """
        self.log_actions = log_actions
        self.action_history = []
        
        if log_actions:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger(__name__)
    
    def execute_action(
        self,
        action_id: int,
        action_name: str,
        behaviour_type: str,
        confidence: float,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Execute an action based on the predicted behaviour.
        
        Args:
            action_id: Unique action identifier
            action_name: Name/description of the action
            behaviour_type: Type of behaviour ('automated' or 'optimising')
            confidence: Confidence score of the prediction
            context: Additional context information
            
        Returns:
            Dictionary with execution results
        """
        timestamp = datetime.now().isoformat()
        
        action_result = {
            'timestamp': timestamp,
            'action_id': action_id,
            'action_name': action_name,
            'behaviour_type': behaviour_type,
            'confidence': confidence,
            'status': 'executed',
            'context': context or {}
        }
        
        # Log the action
        if self.log_actions:
            self.logger.info(
                f"Executing action: {action_name} (ID: {action_id}) | "
                f"Behaviour: {behaviour_type} | Confidence: {confidence:.2f}"
            )
        
        # Store in history
        self.action_history.append(action_result)
        
        # Execute specific logic based on behaviour type
        if behaviour_type == 'automated':
            action_result['execution_details'] = self._handle_automated_behaviour(
                action_id, action_name, context
            )
        elif behaviour_type == 'optimising':
            action_result['execution_details'] = self._handle_optimising_behaviour(
                action_id, action_name, context
            )
        else:
            action_result['status'] = 'skipped'
            action_result['reason'] = 'normal behaviour - no action required'
        
        return action_result
    
    def _handle_automated_behaviour(
        self,
        action_id: int,
        action_name: str,
        context: Optional[Dict]
    ) -> Dict:
        """
        Handle automated behaviour patterns.
        
        Args:
            action_id: Action identifier
            action_name: Action name
            context: Additional context
            
        Returns:
            Execution details
        """
        details = {
            'type': 'automation',
            'action': action_name,
            'strategy': 'pattern_repetition'
        }
        
        # Automation strategies based on action type
        if 'light' in action_name.lower():
            details['device'] = 'lighting_system'
            details['operation'] = 'auto_schedule'
        elif 'temperature' in action_name.lower() or 'thermostat' in action_name.lower():
            details['device'] = 'hvac_system'
            details['operation'] = 'auto_climate_control'
        elif 'security' in action_name.lower():
            details['device'] = 'security_system'
            details['operation'] = 'auto_arm_disarm'
        elif 'appliance' in action_name.lower():
            details['device'] = 'appliance_control'
            details['operation'] = 'auto_power_management'
        else:
            details['device'] = 'general_automation'
            details['operation'] = 'pattern_based_control'
        
        if self.log_actions:
            self.logger.info(f"Automated behaviour handled: {details}")
        
        return details
    
    def _handle_optimising_behaviour(
        self,
        action_id: int,
        action_name: str,
        context: Optional[Dict]
    ) -> Dict:
        """
        Handle optimising behaviour patterns.
        
        Args:
            action_id: Action identifier
            action_name: Action name
            context: Additional context
            
        Returns:
            Execution details
        """
        details = {
            'type': 'optimization',
            'action': action_name,
            'strategy': 'efficiency_improvement'
        }
        
        # Optimization strategies based on action type
        if 'energy' in action_name.lower() or 'power' in action_name.lower():
            details['objective'] = 'energy_efficiency'
            details['optimization'] = 'reduce_consumption'
        elif 'comfort' in action_name.lower():
            details['objective'] = 'user_comfort'
            details['optimization'] = 'balance_comfort_efficiency'
        elif 'cost' in action_name.lower():
            details['objective'] = 'cost_reduction'
            details['optimization'] = 'peak_shaving'
        elif 'schedule' in action_name.lower():
            details['objective'] = 'schedule_optimization'
            details['optimization'] = 'adaptive_scheduling'
        else:
            details['objective'] = 'general_optimization'
            details['optimization'] = 'adaptive_control'
        
        if self.log_actions:
            self.logger.info(f"Optimising behaviour handled: {details}")
        
        return details
    
    def get_action_history(
        self,
        behaviour_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Get action history with optional filtering.
        
        Args:
            behaviour_type: Filter by behaviour type
            limit: Maximum number of records to return
            
        Returns:
            List of action records
        """
        history = self.action_history
        
        if behaviour_type:
            history = [
                action for action in history
                if action['behaviour_type'] == behaviour_type
            ]
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def get_action_statistics(self) -> Dict:
        """
        Get statistics about executed actions.
        
        Returns:
            Dictionary with action statistics
        """
        if not self.action_history:
            return {
                'total_actions': 0,
                'automated_actions': 0,
                'optimising_actions': 0,
                'average_confidence': 0.0
            }
        
        total = len(self.action_history)
        automated = sum(
            1 for a in self.action_history
            if a['behaviour_type'] == 'automated'
        )
        optimising = sum(
            1 for a in self.action_history
            if a['behaviour_type'] == 'optimising'
        )
        avg_confidence = sum(
            a['confidence'] for a in self.action_history
        ) / total
        
        return {
            'total_actions': total,
            'automated_actions': automated,
            'optimising_actions': optimising,
            'normal_actions': total - automated - optimising,
            'average_confidence': avg_confidence,
            'automated_percentage': (automated / total) * 100,
            'optimising_percentage': (optimising / total) * 100
        }
    
    def clear_history(self):
        """Clear the action history."""
        self.action_history = []
        if self.log_actions:
            self.logger.info("Action history cleared")
