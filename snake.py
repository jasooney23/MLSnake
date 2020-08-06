'''Main snake game'''
import random
import pygame

pygame.init()
screen = pygame.display.set_mode((750, 750))

running = True
alive = True

snake_list = [[3, 7], [2, 7], [1, 7]]
food = [11, 7]
snake_size = 50
current_direction = "RIGHT"

clock = pygame.time.Clock()


def draw_snake():
    for j in range(0, len(snake_list)):
        if j == 0:
            pygame.draw.rect(screen, (200, 200, 200),
                             (snake_list[j][0] * snake_size, snake_list[j][1] * snake_size, snake_size, snake_size))
        else:
            pygame.draw.rect(screen, (255, 255, 255),
                             (snake_list[j][0] * snake_size, snake_list[j][1] * snake_size, snake_size, snake_size))


def draw_food():
    pygame.draw.rect(screen, (255, 0, 0),
                     (food[0] * snake_size, food[1] * snake_size, snake_size, snake_size))


def move_snake(direction):
    first_segment = snake_list[0]
    if direction == "UP":
        if first_segment[0] == food[0] and first_segment[1] - 1 == food[1]:
            eat_food()
        else:
            last_segment = snake_list.pop(len(snake_list) - 1)
            last_segment[0] = first_segment[0]
            last_segment[1] = first_segment[1] - 1
            snake_list.insert(0, last_segment)

    elif direction == "DOWN":
        if first_segment[0] == food[0] and first_segment[1] + 1 == food[1]:
            eat_food()
        else:
            last_segment = snake_list.pop(len(snake_list) - 1)
            last_segment[0] = first_segment[0]
            last_segment[1] = first_segment[1] + 1
            snake_list.insert(0, last_segment)

    elif direction == "LEFT":
        if first_segment[0] - 1 == food[0] and first_segment[1] == food[1]:
            eat_food()
        else:
            last_segment = snake_list.pop(len(snake_list) - 1)
            last_segment[0] = first_segment[0] - 1
            last_segment[1] = first_segment[1]
            snake_list.insert(0, last_segment)

    elif direction == "RIGHT":
        if first_segment[0] + 1 == food[0] and first_segment[1] == food[1]:
            eat_food()
        else:
            last_segment = snake_list.pop(len(snake_list) - 1)
            last_segment[0] = first_segment[0] + 1
            last_segment[1] = first_segment[1]
            snake_list.insert(0, last_segment)


def eat_food():
    snake_list.insert(0, [food[0], food[1]])
    food[0] = random.randint(0, 14)
    food[1] = random.randint(0, 14)


while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if pygame.key.get_pressed()[pygame.K_UP]:
        current_direction = "UP"
    elif pygame.key.get_pressed()[pygame.K_DOWN]:
        current_direction = "DOWN"
    elif pygame.key.get_pressed()[pygame.K_LEFT]:
        current_direction = "LEFT"
    elif pygame.key.get_pressed()[pygame.K_RIGHT]:
        current_direction = "RIGHT"

    move_snake(current_direction)

    if snake_list[0][0] < 0 or snake_list[0][0] >= 15 or snake_list[0][1] < 0 or snake_list[0][1] >= 15:
        alive = False

    for i in range(0, len(snake_list)):
        if i != 0:
            if snake_list[0] == snake_list[i]:
                alive = False

    if alive:
        screen.fill((0, 0, 0))
        draw_snake()
        draw_food()
        pygame.display.flip()

    clock.tick(5)

pygame.quit()
