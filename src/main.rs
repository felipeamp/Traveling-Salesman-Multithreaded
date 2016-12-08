#[macro_use]
extern crate lazy_static;

use std::cell::UnsafeCell;
use std::collections::VecDeque;
use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use std::sync::{Mutex, Condvar};
use std::thread;


////////////////////////////////////////////////////////////
// Partial Path
////////////////////////////////////////////////////////////

#[derive(Clone, Debug)]
struct PartialPath {
    order_visited: Vec<usize>,
    visited: Vec<bool>,
    cost_so_far: u64,
}

impl PartialPath {
    fn new() -> PartialPath {
        PartialPath {
            order_visited: Vec::new(),
            visited: vec![false; NUM_CITIES],
            cost_so_far: 0,
        }
    }

    fn push(&mut self, new_city: usize, dist_matrix: &[[u64; NUM_CITIES]; NUM_CITIES]) {
        if new_city > self.visited.len() {
            panic!("New city ({}) is larger than last city index ({}).",
                   new_city,
                   self.visited.len());
        }
        if self.visited[new_city] {
            panic!("New city ({}) has already been visited.", new_city);
        }
        self.cost_so_far += dist_matrix[new_city][*self.order_visited.last().unwrap()];
        self.order_visited.push(new_city);
        self.visited[new_city] = true;
    }

    fn close_tour(&mut self, dist_matrix: &[[u64; NUM_CITIES]; NUM_CITIES]) {
        self.cost_so_far += dist_matrix[TOUR_START][*self.order_visited.last().unwrap()];
        self.order_visited.push(TOUR_START);
    }
}


////////////////////////////////////////////////////////////
// Best Found
////////////////////////////////////////////////////////////

#[derive(Debug)]
struct BestFound {
    cost: UnsafeCell<u64>,
    tour: Mutex<Vec<usize>>,
}

impl BestFound {
    fn new() -> BestFound {
        BestFound{
            cost: UnsafeCell::new(u64::max_value()),
            tour: Mutex::new(Vec::new()),
        }
    }

    fn update_if_best(&self, new_best_path: PartialPath) {
        let mut tour = self.tour.lock().unwrap();
        unsafe {
            let best_cost = self.cost.get();
            if new_best_path.cost_so_far < *best_cost {
                *tour = new_best_path.order_visited.clone();
                *best_cost = new_best_path.cost_so_far;
            }
        }
    }

    fn read_best(&self) -> u64 {
        unsafe { *(self.cost.get()) }
    }

    fn print(&self) {
        println!("");
        println!("-----------------------------------------------------");
        println!("Best cost found: {}", BEST_FOUND.read_best());
        println!("");
        println!("The best tour is:");
        let best_tour_guard = BEST_FOUND.tour.lock().unwrap();
        for city_index in (*best_tour_guard).iter() {
            println!("\t {}", city_index + 1);
        }
        println!("-----------------------------------------------------");
        println!("");
    }
}

unsafe impl Send for BestFound { }
unsafe impl Sync for BestFound { }


////////////////////////////////////////////////////////////
// Term
////////////////////////////////////////////////////////////

struct Term {
    new_stack: Mutex<Vec<PartialPath>>,
    stack_is_empty: UnsafeCell<bool>,
    threads_waiting: UnsafeCell<usize>,
    term_cond_var: Condvar,
}

impl Term {
    fn new() -> Term {
        Term {new_stack: Mutex::new(Vec::new()),
              stack_is_empty: UnsafeCell::new(true),
              threads_waiting: UnsafeCell::new(0),
              term_cond_var: Condvar::new(),}
    }

    fn has_thread_waiting(&self) -> bool {
        unsafe { *(self.threads_waiting.get()) > 0 && *(self.stack_is_empty.get()) }
    }

    fn push_stack(&self, thread_stack: &mut Vec<PartialPath>) -> bool {
        if let Ok(mut new_stack_guard) = self.new_stack.try_lock() {
            if self.has_thread_waiting() {
                *new_stack_guard = split_stack(thread_stack);
                unsafe {
                    *(self.stack_is_empty.get()) = false;
                }
                drop(new_stack_guard);
                self.term_cond_var.notify_one();
            }
            return true;
        } else {
            return false;
        }
    }

    fn get_stack(&self, out_stack: &mut Vec<PartialPath>) -> bool {
        let mut new_stack_guard = self.new_stack.lock().unwrap();
        let thread_waiting_p = self.threads_waiting.get();
        unsafe {
            if *thread_waiting_p == NUM_THREADS - 1 {
                *thread_waiting_p += 1;
                println!("ALL THREADS FINISHED!");
                println!("Last thread was {}", thread::current().name().unwrap());
                drop(new_stack_guard);
                self.term_cond_var.notify_one();
                BEST_FOUND.print();
                return true;
            } else {
                *thread_waiting_p += 1;
                println!("{} THREADS WAITING!", *thread_waiting_p);
                println!("Last thread was {}", thread::current().name().unwrap());
                let stack_is_empty_p = self.stack_is_empty.get();
                while *stack_is_empty_p {
                    new_stack_guard = self.term_cond_var.wait(new_stack_guard).unwrap();
                }
                println!("THREAD {} WOKEN UP!", thread::current().name().unwrap());
                if *thread_waiting_p < NUM_THREADS {
                    println!("FOUND NEW STACK!");
                    *out_stack = (*new_stack_guard).clone();
                    *new_stack_guard = Vec::new();
                    *stack_is_empty_p = true;
                    *thread_waiting_p -= 1;
                    return false;
                } else {
                    println!("FINISHING THREAD {}!", thread::current().name().unwrap());
                    drop(new_stack_guard);
                    self.term_cond_var.notify_one();
                    return true;
                }
            }
        }
    }
}

unsafe impl Send for Term { }
unsafe impl Sync for Term { }


////////////////////////////////////////////////////////////
// Static's and Const's
////////////////////////////////////////////////////////////

const FILEPATH_LARGE: &'static str = "att20_d.csv";
const FILEPATH_MEDIUM: &'static str = "att15_d.csv";
const FILEPATH_SMALL: &'static str = "att10_d.csv";
const FILEPATH: &'static str = FILEPATH_MEDIUM;

const NUM_CITIES_LARGE: usize = 20;
const NUM_CITIES_MEDIUM: usize = 15;
const NUM_CITIES_SMALL: usize = 10;
const NUM_CITIES: usize = NUM_CITIES_MEDIUM;

const NUM_THREADS: usize = 8;

const TOUR_START: usize = 0;

lazy_static! {
    static ref BEST_FOUND: BestFound = BestFound::new();
    static ref TERM: Term = Term::new();
}


////////////////////////////////////////////////////////////
// Functions
////////////////////////////////////////////////////////////

fn load_dist_matrix(path: &Path) -> [[u64; NUM_CITIES]; NUM_CITIES] {
    // Open the path in read-only mode, returns `io::Result<File>`
    let mut file = match File::open(path) {
        // The `description` method of `io::Error` returns a string that
        // describes the error
        Err(why) => panic!("couldn't open {}: {}", path.to_str().unwrap(),
                                                   why.description()),
        Ok(file) => file,
    };

    // Read the file contents into a string, returns `io::Result<usize>`
    let mut s = String::new();
    match file.read_to_string(&mut s) {
        Err(why) => panic!("couldn't read {}: {}", path.to_str().unwrap(),
                                                   why.description()),
        Ok(_) => (),
    };

    // Split s in NUM_CITIES lines, remove leading/trailing whitespace from each and split
    // in NUM_CITIES buckets
    let mut dist_matrix: [[u64; NUM_CITIES]; NUM_CITIES] = [[0; NUM_CITIES]; NUM_CITIES];
    let split = s.split("\n");
    for (row_index, line) in split.enumerate() {
        // DEBUG:
        // println!("{}", line);
        if line.len() == 0 {
            continue;
        }
        for (column_index, word) in line.trim().split(",").enumerate() {
            dist_matrix[row_index][column_index] = word.parse().unwrap();
        }
    }
    dist_matrix
}

fn bfs(dist_matrix: &[[u64; NUM_CITIES]; NUM_CITIES]) -> VecDeque<PartialPath> {
    let mut starting_path = PartialPath::new();
    starting_path.order_visited.push(TOUR_START);
    starting_path.visited[TOUR_START] = true;

    let mut ret: VecDeque<PartialPath> = VecDeque::new();
    ret.push_back(starting_path);

    while ret.len() < NUM_THREADS {
        let curr_path = ret.pop_front().unwrap();
        for next_city in 0..NUM_CITIES {
            if curr_path.visited[next_city] {
                continue;
            }
            let mut new_path = curr_path.clone();
            new_path.push(next_city, dist_matrix);
            ret.push_back(new_path);
        }
    }
    ret
}

fn divide_workload(work_load: VecDeque<PartialPath>) -> Vec<Vec<PartialPath>> {
    let mut ret: Vec<Vec<PartialPath>> = Vec::new();

    let work_size = work_load.len();
    let work_per_thread: f64 = (work_size as f64) / (NUM_THREADS as f64);

    let mut work_sent: usize = 0;
    let mut work_sent_frac: f64 = 0.0;

    for _ in 0..NUM_THREADS {
        work_sent_frac += work_per_thread;
        let mut curr_vec_work: Vec<PartialPath> = Vec::new();
        let work_range_limit: usize = work_sent_frac.floor() as usize;
        for index in work_sent..work_range_limit {
            curr_vec_work.push(work_load[index].clone());
        }
        ret.push(curr_vec_work);
        work_sent = work_sent_frac.floor() as usize;
    }
    if work_sent != work_size {
        ret.last_mut().unwrap().push(work_load[work_size - 1].clone());
    }
    ret
}

fn split_stack(stack: &mut Vec<PartialPath>) -> Vec<PartialPath> {
    let mut new_stack_1: Vec<PartialPath> = Vec::new();
    let mut new_stack_2: Vec<PartialPath> = Vec::new();
    for (index, path) in stack.into_iter().enumerate() {
        if index % 2 == 0 {
            new_stack_2.push(path.clone());
        } else {
            new_stack_1.push(path.clone());
        }
    }
    *stack = new_stack_1;
    new_stack_2
}

fn dfs(mut stack: Vec<PartialPath>, dist_matrix: [[u64; NUM_CITIES]; NUM_CITIES],
       best_found: &BestFound, term: &Term) {
    loop {
        while !stack.is_empty() {
            if term.has_thread_waiting() && stack.len() > 2 {
                if term.push_stack(&mut stack) {
                    continue;
                }
            }
            let mut curr_path = stack.pop().unwrap();
            if curr_path.order_visited.len() == NUM_CITIES {
                // This path is complete. Let's close the tour and update best_found
                curr_path.close_tour(&dist_matrix);
                println!("Completed a path with cost = {}", curr_path.cost_so_far);
                if curr_path.cost_so_far < best_found.read_best() {
                    println!("Best path found so far!");
                    best_found.update_if_best(curr_path);
                }
                continue;
            }
            for new_city in 0..NUM_CITIES {
                if curr_path.visited[new_city] {
                    continue;
                }
                let mut new_path = curr_path.clone();
                new_path.push(new_city, &dist_matrix);
                if new_path.cost_so_far < BEST_FOUND.read_best() {
                    stack.push(new_path);
                }
            }
        }
        if term.get_stack(&mut stack) {
            return;
        }
    }
}

fn main() {
    let dist_matrix: [[u64; NUM_CITIES]; NUM_CITIES] = load_dist_matrix(
        &Path::new(".").join("..").join(FILEPATH));
    let work_load = bfs(&dist_matrix);
    let work_per_thread = divide_workload(work_load);

    let mut spawned_threads: Vec<thread::JoinHandle<_>> = Vec::new();
    for thread_number in 0..NUM_THREADS {
        let stack = work_per_thread[thread_number].clone();
        let local_dist_matrix = dist_matrix.clone();
        spawned_threads.push(thread::Builder::new().
            name(format!("child_{}", thread_number + 1).to_string()).spawn(
                move || { dfs(stack, local_dist_matrix, &BEST_FOUND, &TERM);
            }).unwrap());
    }

    for (thread_index, child) in spawned_threads.into_iter().enumerate() {
        let _ = child.join();
        println!("Thread {} has joined!", thread_index + 1);
    }
}
