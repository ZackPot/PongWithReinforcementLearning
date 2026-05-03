import pygame
import numpy as np
from tqdm import trange
import os

pygame.init()

# Font used to render the text
font20 = pygame.font.Font('freesansbold.ttf', 20)

# RGB values of standard colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)

# Basic parameters of the screen
WIDTH, HEIGHT = 900, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong")

clock = pygame.time.Clock()
FPS = 30

# Striker class
class Striker:
    def __init__(self, posx, posy, width, height, speed, color):
        self.posx = posx
        self.posy = posy
        self.width = width
        self.height = height
        self.speed = speed
        self.color = color
        self.geekRect = pygame.Rect(posx, posy, width, height)

    def display(self):
        pygame.draw.rect(screen, self.color, self.geekRect)

    def update(self, yFac):
        self.posy = self.posy + self.speed * yFac

        # Restricting the striker bounds
        if self.posy <= 0:
            self.posy = 0
        elif self.posy + self.height >= HEIGHT:
            self.posy = HEIGHT - self.height

        self.geekRect = pygame.Rect(self.posx, self.posy, self.width, self.height)

    def displayScore(self, text, score, x, y, color):
        text_surface = font20.render(text + str(score), True, color)
        textRect = text_surface.get_rect()
        textRect.center = (x, y)
        screen.blit(text_surface, textRect)

    def getRect(self):
        return self.geekRect

# Ball class
class Ball:
    def __init__(self, posx, posy, radius, speed, color):
        self.posx = posx
        self.posy = posy
        self.radius = radius
        self.base_speed = speed # Store initial speed for resets
        self.speed = speed
        self.color = color
        self.xFac = 1
        self.yFac = -1 # Fixed: Changed from string to integer
        self.firstTime = 1

    def display(self):
        self.ball = pygame.draw.circle(screen, self.color, (int(self.posx), int(self.posy)), self.radius)

    def update(self):
        self.posx += self.speed * self.xFac
        self.posy += self.speed * self.yFac

        # Reflection on top/bottom walls
        if self.posy <= 0 or self.posy >= HEIGHT:
            self.yFac *= -1

        # Scoring logic
        if self.posx <= 0 and self.firstTime:
            self.firstTime = 0
            return 1
        elif self.posx >= WIDTH and self.firstTime:
            self.firstTime = 0
            return -1
        else:
            return 0

    def reset(self):
        self.posx = WIDTH // 2
        self.posy = HEIGHT // 2
        self.xFac *= -1
        self.speed = self.base_speed # Reset speed for the new round
        self.firstTime = 1

    def hit(self):
        self.xFac *= -1
        self.speed += 0.5 # INCREASE SPEED: Increments by 0.5 on every paddle hit

    def getRect(self):
        # Create a Rect for the ball to handle collisions properly
        return pygame.Rect(self.posx - self.radius, self.posy - self.radius, self.radius * 2, self.radius * 2)

# Game Manager
def main():
    global reward, state, action, q_table
    running = True

    geek1 = Striker(20, 0, 10, 100, 10, GREEN)
    geek2 = Striker(WIDTH - 30, 0, 10, 100, 10, GREEN)
    ball = Ball(WIDTH // 2, HEIGHT // 2, 7, 7, WHITE)

    listOfGeeks = [geek1, geek2]

    geek1Score, geek2Score = 0, 0
    geek1YFac, geek2YFac = 0, 0

    # ----------------------- Reinforcement Learning ----------------------------

    alpha, gamma, epsilon = 0.1, 0.9, 0.1

    bins = 10
    vy_bins = np.linspace(0, HEIGHT, bins)
    vx_bins = np.linspace(0, WIDTH, bins)
    player_pos_bins = np.linspace(0, HEIGHT, bins)

    q_table = np.zeros((bins + 1, bins + 1, bins + 1, 3))  # bins = num of states, 3 = up, down or NOOP action

    if os.path.exists("pong_q_table.npy"):
        q_table = np.load("pong_q_table.npy")
        print("Loaded existing Q-table.")
    else:
        q_table = np.zeros((bins + 1, bins + 1, bins + 1, 3))
        print("Created new Q-table.")

    # ----------------------- Game Logic ----------------------------

    for _ in trange(0, 1000, desc="Episode"):
        epsilon = max(0.01, epsilon * 0.999)
        ball.reset()

        for _ in trange(1000, desc="Frame"):
            vy_discrete = np.digitize(ball.posy, vy_bins)
            vx_discrete = np.digitize(ball.posx, vx_bins)
            player_pos = np.digitize(geek2.posy, player_pos_bins)
            state = (vy_discrete, vx_discrete, player_pos)

            if np.random.uniform(0, 1) < epsilon:
                action = np.random.choice([0, 1, 2])
            else:
                action = np.argmax(q_table[state])

            action_map = {0:0, 1:1, 2:-1}
            geek2YFac = action_map[action]

            screen.fill(BLACK)

            if ball.posy > geek1.posy + (geek1.height / 2):
                geek1YFac = 1
            elif ball.posy < geek1.posy + (geek1.height / 2):
                geek1YFac = -1


            # Collision detection with guard to prevent "sticking"
            for geek in listOfGeeks:
                if pygame.Rect.colliderect(ball.getRect(), geek.getRect()):
                    if (geek.posx < WIDTH//2 and ball.xFac < 0) or (geek.posx > WIDTH//2 and ball.xFac > 0):
                        ball.hit()

            geek1.update(geek1YFac)
            geek2.update(geek2YFac)
            point = ball.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            if point == -1: # Fixed string comparison
                geek1Score += 1
            elif point == 1:
                geek2Score += 1

            if point:
                ball.reset()

            geek1.display()
            geek2.display()
            ball.display()

            geek1.displayScore("Geek_1 : ", geek1Score, 100, 20, WHITE)
            geek2.displayScore("Geek_2 : ", geek2Score, WIDTH - 100, 20, WHITE)

            hit = pygame.Rect.colliderect(ball.getRect(), geek2.getRect())
            if hit:
                reward = 5
            elif point == -1:
                reward = -20
            else:
                dist = abs(geek2.posy + (geek2.height / 2) - ball.posy)
                reward = 0.1 if dist < 50 else 0

            clock.tick(FPS)
            pygame.display.update()

            vy_discrete = np.digitize(ball.posy, vy_bins)
            vx_discrete = np.digitize(ball.posx, vx_bins)
            player_pos = np.digitize(geek2.posy, player_pos_bins)
            next_state = (vy_discrete, vx_discrete, player_pos)

            best_next_action = np.max(q_table[next_state])
            q_table[state][action] += alpha * (reward + gamma * best_next_action - q_table[state][action])

main()
np.save('q_table.npy', q_table)
pygame.quit()
