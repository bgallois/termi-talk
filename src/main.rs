use either::Either;
use indexmap::IndexMap;
use mistralrs::MessageContent;
use mistralrs::{
    MistralRs, NormalRequest, Request, RequestMessage, ResponseOk, Result, SamplingParams,
};
use ratatui::{
    crossterm::event::{self, Event, KeyCode, KeyEventKind},
    layout::{Constraint, Layout, Position},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span, Text},
    widgets::{Block, List, ListItem, ListState, Paragraph},
    DefaultTerminal, Frame,
};
use std::fmt;
use std::sync::Arc;
use tokio::sync::mpsc::channel;

mod inference;
use inference::*;

fn main() -> Result<()> {
    let terminal = ratatui::init();
    let app_result = App::new().run(terminal);
    ratatui::restore();
    app_result
}

struct App {
    input: String,
    character_index: usize,
    input_mode: InputMode,
    messages: Vec<(Who, String)>,
    messages_state: ListState,
    model: Arc<MistralRs>,
    context: Vec<IndexMap<String, MessageContent>>,
    context_len: usize,
}

#[derive(PartialEq)]
enum InputMode {
    Normal,
    Editing,
    Generating,
}

#[derive(Clone)]
enum Who {
    Me,
    Assistant,
    Empty,
}

impl fmt::Display for Who {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Who::Me => write!(f, "Me"),
            Who::Assistant => write!(f, "QS"),
            Who::Empty => write!(f, "  "),
        }
    }
}

impl App {
    fn new() -> Self {
        Self {
            input: String::new(),
            input_mode: InputMode::Editing,
            messages: Vec::new(),
            messages_state: ListState::default(),
            character_index: 0,
            model: load_model().unwrap(),
            context: vec![IndexMap::from([
                ("role".to_string(), Either::Left("system".to_string())),
                (
                    "content".to_string(),
                    Either::Left(SYSTEM_PROMPT.to_string()),
                ),
            ])],
            context_len: SYSTEM_PROMPT.len(),
        }
    }

    fn move_cursor_left(&mut self) {
        let cursor_moved_left = self.character_index.saturating_sub(1);
        self.character_index = self.clamp_cursor(cursor_moved_left);
    }

    fn move_cursor_right(&mut self) {
        let cursor_moved_right = self.character_index.saturating_add(1);
        self.character_index = self.clamp_cursor(cursor_moved_right);
    }

    fn enter_char(&mut self, new_char: char) {
        let index = self.byte_index();
        self.input.insert(index, new_char);
        self.move_cursor_right();
    }

    fn byte_index(&self) -> usize {
        self.input
            .char_indices()
            .map(|(i, _)| i)
            .nth(self.character_index)
            .unwrap_or(self.input.len())
    }

    fn delete_char(&mut self) {
        let is_not_cursor_leftmost = self.character_index != 0;
        if is_not_cursor_leftmost {
            let current_index = self.character_index;
            let from_left_to_current_index = current_index - 1;

            let before_char_to_delete = self.input.chars().take(from_left_to_current_index);
            let after_char_to_delete = self.input.chars().skip(current_index);

            self.input = before_char_to_delete.chain(after_char_to_delete).collect();
            self.move_cursor_left();
        }
    }

    fn clamp_cursor(&self, new_cursor_pos: usize) -> usize {
        new_cursor_pos.clamp(0, self.input.chars().count())
    }

    fn reset_cursor(&mut self) {
        self.character_index = 0;
    }

    fn submit_message(&mut self) {
        self.messages.push((Who::Me, self.input.clone()));

        self.context.push(IndexMap::from([
            ("role".to_string(), Either::Left("user".to_string())),
            ("content".to_string(), Either::Left(self.input.clone())),
        ]));
        self.context_len += self.input.len();

        let (tx, mut rx) = channel(10_000);
        let request = Request::Normal(NormalRequest {
            messages: RequestMessage::Chat(self.context.clone()),
            sampling_params: SamplingParams {
                temperature: Some(1.5),
                top_k: Some(50),
                top_p: Some(0.7),
                ..SamplingParams::deterministic()
            },
            response: tx,
            return_logprobs: false,
            is_streaming: false,
            id: 0,
            constraint: mistralrs::Constraint::None,
            suffix: None,
            adapters: None,
            tools: None,
            tool_choice: None,
            logits_processors: None,
        });
        self.model
            .get_sender()
            .unwrap()
            .blocking_send(request)
            .unwrap();

        let response = rx.blocking_recv().unwrap().as_result().unwrap();
        if let ResponseOk::Done(c) = response {
            self.messages.push((
                Who::Assistant,
                c.choices[0].message.content.as_ref().unwrap().to_string(),
            ));
            self.context.push(IndexMap::from([
                ("role".to_string(), Either::Left("assistant".to_string())),
                (
                    "content".to_string(),
                    Either::Left(
                        c.choices[0]
                            .message
                            .content
                            .as_ref()
                            .unwrap()
                            .chars()
                            .filter(|c| c.is_alphanumeric())
                            .collect(),
                    ),
                ),
            ]));
            self.context_len += c.choices[0]
                .message
                .content
                .as_ref()
                .unwrap()
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
                .len();
        }

        while self.context_len > 1000 {
            let question = self.context.remove(1);
            let answer = self.context.remove(1);
            self.context_len -= question
                .get("content")
                .unwrap()
                .clone()
                .left()
                .unwrap()
                .len()
                + answer.get("content").unwrap().clone().left().unwrap().len();
        }

        self.input.clear();
        self.reset_cursor();
    }

    fn run(mut self, mut terminal: DefaultTerminal) -> Result<()> {
        loop {
            terminal.draw(|frame| self.draw(frame))?;

            if self.input_mode == InputMode::Generating {
                self.submit_message();
                self.input_mode = InputMode::Editing;
                terminal.draw(|frame| self.draw(frame))?;
            }

            if let Event::Key(key) = event::read()? {
                match self.input_mode {
                    InputMode::Normal => match key.code {
                        KeyCode::Char('e') => {
                            self.input_mode = InputMode::Editing;
                        }
                        KeyCode::Char('q') => {
                            return Ok(());
                        }
                        _ => {}
                    },
                    InputMode::Editing if key.kind == KeyEventKind::Press => match key.code {
                        KeyCode::Enter => {
                            self.input_mode = InputMode::Generating;
                        }
                        KeyCode::Char(to_insert) => self.enter_char(to_insert),
                        KeyCode::Backspace => self.delete_char(),
                        KeyCode::Left => self.move_cursor_left(),
                        KeyCode::Right => self.move_cursor_right(),
                        KeyCode::Esc => self.input_mode = InputMode::Normal,
                        _ => {}
                    },
                    InputMode::Editing => {}
                    InputMode::Generating => {}
                }
            }
        }
    }

    fn draw(&mut self, frame: &mut Frame) {
        let vertical = Layout::vertical([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(3),
        ]);
        let [help_area, messages_area, input_area] = vertical.areas(frame.area());

        let (msg, style) = match self.input_mode {
            InputMode::Normal => (
                vec![
                    "Press ".into(),
                    "q".bold(),
                    " to exit, ".into(),
                    "e".bold(),
                    " to start editing.".bold(),
                ],
                Style::default().add_modifier(Modifier::RAPID_BLINK),
            ),
            InputMode::Editing => (
                vec![
                    "Press ".into(),
                    "Esc".bold(),
                    " to stop editing, ".into(),
                    "Enter".bold(),
                    " to record the message".into(),
                ],
                Style::default(),
            ),
            InputMode::Generating => (
                vec!["I'm thinking".bold(), " WAIT".bold()],
                Style::default().fg(Color::Red),
            ),
        };
        let text = Text::from(Line::from(msg)).patch_style(style);
        let help_message = Paragraph::new(text);
        frame.render_widget(help_message, help_area);

        let input = Paragraph::new(self.input.as_str())
            .style(match self.input_mode {
                InputMode::Normal => Style::default(),
                InputMode::Editing => Style::default().fg(Color::Yellow),
                InputMode::Generating => Style::default(),
            })
            .block(Block::bordered().title("Input"));
        frame.render_widget(input, input_area);
        match self.input_mode {
            // Hide the cursor. `Frame` does this by default, so we don't need to do anything here
            InputMode::Normal => {}
            InputMode::Generating => {}

            // Make the cursor visible and ask ratatui to put it at the specified coordinates after
            // rendering
            #[allow(clippy::cast_possible_truncation)]
            InputMode::Editing => frame.set_cursor_position(Position::new(
                // Draw the cursor at the current position in the input field.
                // This position is can be controlled via the left and right arrow key
                input_area.x + self.character_index as u16 + 1,
                // Move one line down, from the border to the input line
                input_area.y + 1,
            )),
        }

        let messages: Vec<ListItem> = self
            .messages
            .iter()
            .flat_map(|m| {
                let lines = wrap_text(m.1.clone(), frame.area().width.into());
                lines.into_iter().enumerate().map(move |(i, line)| {
                    let who = if i != 0 { Who::Empty } else { m.0.clone() };
                    let content = Line::from(Span::raw(format!("{} {}", who, line)));
                    match m.0 {
                        Who::Me => ListItem::new(content).red(),
                        Who::Assistant => ListItem::new(content).green(),
                        Who::Empty => ListItem::new(content).green(),
                    }
                })
            })
            .collect();

        let messages = List::new(messages).block(Block::bordered().title("Messages"));
        self.messages_state.select_last();
        frame.render_stateful_widget(messages, messages_area, &mut self.messages_state);
    }
}
