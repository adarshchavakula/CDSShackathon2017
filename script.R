library(jsonlite)
install.packages('jsonlite')

df1 <- read_json('enron-sentences.json', simplifyVector = F)
purrr::map('enron-sentences.json', jsonlite::fromJSON)

install.packages('ndjson')
library(ndjson)
bitly01 <- ndjson::stream_in('enron-sentences.json')

df1 <- read.csv('enron_emails.csv')


### bad words analysis ###
processed <- read.csv('processed_enron.csv')
bad_words <- c('fraud', 'fraudulent',	'scam',	'embezzlement',	'forgery',	'swindle',	'fraudsters',	'bribery',	'theft',	'defrauded',	'corruption',	'suspicous',	'suspect',	'amiss',	'fishy',	'alarmed',	'unexplained',	'strange',	'mysterious',	'untoward',	'police',	'policeman',	'constable',	'patrol',	'detective',	'arrested',	'apprehended',	'trooper',	'arrest',	'arresting',	'cop')
bd <- processed
bd$content_proc <- str_replace_all(bd$content, "[[:punct:]]", "")
bd$content_proc <- strsplit(bd$content_proc, " ")

library(stringr)
test <- "I am a fraud!"
x <- "a1~!@#$%^&*(){}_+:\"<>?,./;'[]-=" #or whatever
test <- str_replace_all(test, "[[:punct:]]", "")




######## graphing ######
library(igraph)
library(dplyr)
send_rec_back <- send_rec
send_rec <- read.csv('graph_input_years.csv')
final_graph_input <- send_rec[c('From.x', 'To.x', 'Date.x', 'from_department.x','to_department.x', 'receivers')]
names(final_graph_input) <- c('from', 'to','date','from_dept', 'to_dept', 'receivers')
final_graph_input$year <- as.numeric(substring(final_graph_input$date,1,4))











senders_agg <- group_by(final_graph_input, year, from_dept)
senders_agg <- summarise(senders_agg, sent = n())
senders_agg <- senders_agg[(order(-senders_agg$sent)),]
senders_agg$cumsum <- cumsum(senders_agg$sent)
senders_agg$perc <- senders_agg$cumsum/sum(senders_agg$sent)
imp_senders <- senders_agg[senders_agg$perc <= 0.50,]
tester <- final_graph_input
#tester <- final_graph_input[final_graph_input$from %in% imp_senders$from,]
#tester <- tester[tester$to %in% imp_senders$from,]  
tester$year <- as.numeric(tester$year)
tester <- group_by(tester, from_dept, to_dept, year)
tester <- summarise(tester, weight =n())
#tester$from <- as.numeric(as.factor(tester$from))
#tester$to <- as.numeric(as.factor(tester$to))
tester1999 <- tester[tester$year %in% c(2000,2001),]
inputToGraph <- tester1999[c('from_dept','to_dept')]
inputToGraph$from_dept <- as.character(inputToGraph$from_dept)
inputToGraph$to_dept <- as.character(inputToGraph$to_dept)
library(igraph)

#clusters    = g.clusters()
#giant       = clusters.giant() ## using the biggest component as an example, you can use the others here.
#communities = giant.community_spinglass()

#g <- graph_from_edgelist(as.matrix(inputToGraph), directed = T)
g <- graph_from_data_frame(inputToGraph, directed = T)
coms <- spinglass.community(g)
dg <- decompose.graph(g)
coms <- spinglass.community(dg[[1]])

#cl <- clusters(g)
#lcc = induced.subgraph(g, V(g)[which(cl$membership == which.max(cl$csize))])

#coms <- spinglass.community(lcc)

# Plot network
par(mar = c(0,0,2,0))
plot(coms, g, 
     vertex.label=NA, 
     layout = layout.fruchterman.reingold,
     vertex.size = 1,
     edge.arrow.size=0.01,
     arrow.size = 0.5,
     main = 'Department [2001-2002]'
)

net <- 
  ceb <- cluster_edge_betweenness(net)
