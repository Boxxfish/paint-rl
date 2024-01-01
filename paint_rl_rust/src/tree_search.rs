use std::collections::HashMap;

use tch::{nn::Module, Tensor};

use crate::{discs_to_strokes, env::SimCanvasEnv, sample_model};

/// An action that can be performed from a node.
/// Used to query child nodes.
/// Corresponds to (mid_x, mid_y, end_x, end_y, pen_up).
type TsAction = (u32, u32, u32, u32, bool);

/// A tree search node.
#[derive(Clone)]
pub struct TsNode {
    pub ret: f32,
    /// Due to the size of the action space, we lazily generate children.
    pub children: HashMap<TsAction, TsNode>,
}

impl TsNode {
    /// Creates a new tree search node.
    pub fn new(ret: f32) -> Self {
        Self {
            ret,
            children: HashMap::new(),
        }
    }

    /// Recursively expands a node, updating return estimates.
    /// Returns the new return estimate for this node.
    pub fn expand(
        &mut self,
        prev_score: f32,
        obs: &Tensor,
        env: &mut SimCanvasEnv,
        p_net: &tch::CModule,
        v_net: &tch::CModule,
        rew_net: &tch::CModule,
    ) -> f32 {
        // Sample action with stroke probabilities
        let (action_mid, action_end, action_pen_down) = sample_model(p_net, &obs.unsqueeze(0));
        let cont_action = discs_to_strokes(&action_mid, &action_end).squeeze();
        let (next_obs, _, done, _) = env.step(&cont_action, action_pen_down[0]);

        // Compute reward
        let obs_select = Tensor::from_slice(&[4, 5, 6, 2]);
        let score = rew_net
            .forward(&next_obs.index_select(0, &obs_select).unsqueeze(0))
            .squeeze()
            .double_value(&[]) as f32;
        let rew = score - prev_score;
        let ret = rew + v_net.forward(&next_obs.unsqueeze(0)).squeeze().double_value(&[]) as f32;

        // If we've reached the end, return just the reward
        if done {
            return rew;
        }

        // Expand child
        let ts_action = (
            cont_action.double_value(&[0]) as u32,
            cont_action.double_value(&[1]) as u32,
            cont_action.double_value(&[2]) as u32,
            cont_action.double_value(&[3]) as u32,
            action_pen_down[0] == 1,
        );
        let child_ret = if let Some(child) = self.children.get_mut(&ts_action) {
            child.expand(score, &next_obs, env, p_net, v_net, rew_net)
        } else {
            // Add child
            let child = TsNode::new(ret);
            self.children.insert(ts_action, child);
            ret
        };
        self.ret = self.ret.max(child_ret);
        self.ret
    }

    /// Returns the action with the highest return.
    pub fn get_best_action(&self) -> &TsAction {
        self.children
            .iter()
            .max_by(|(_, x), (_, y)| x.ret.total_cmp(&y.ret))
            .unwrap()
            .0
    }

    /// Returns a list of best actions from this node to the end.
    pub fn get_best_actions(&self) -> Vec<TsAction> {
        if self.children.is_empty() {
            return Vec::new();
        }

        let (best_action, best_child) = self
            .children
            .iter()
            .max_by(|(_, x), (_, y)| x.ret.total_cmp(&y.ret))
            .unwrap();
        let mut actions = vec![*best_action];
        actions.append(&mut best_child.get_best_actions());
        actions
    }
}
